import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast

from rich.console import Console
from rich.markup import escape

import vision_agent.tools as T
from vision_agent.agent import AgentCoder, AgentPlanner
from vision_agent.agent.agent_utils import (
    DefaultImports,
    add_media_to_chat,
    capture_media_from_exec,
    convert_message_to_agentmessage,
    extract_tag,
    format_feedback,
    format_plan_v2,
    print_code,
    strip_function_calls,
)
from vision_agent.agent.types import AgentMessage, CodeContext, PlanContext
from vision_agent.agent.vision_agent_coder_prompts_v2 import CODE, FIX_BUG, TEST
from vision_agent.agent.vision_agent_planner_v2 import VisionAgentPlannerV2
from vision_agent.lmm import LMM, AnthropicLMM
from vision_agent.lmm.types import Message
from vision_agent.tools.meta_tools import get_diff
from vision_agent.utils.execute import (
    CodeInterpreter,
    CodeInterpreterFactory,
    Execution,
)
from vision_agent.utils.sim import Sim, load_cached_sim

_CONSOLE = Console()


def format_code_context(
    code_context: CodeContext,
) -> str:
    return f"<final_code>{code_context.code}</final_code>\n<final_test>{code_context.test}</final_test>"


def retrieve_tools(
    plan: List[str],
    tool_recommender: Sim,
) -> str:
    tool_docs = []
    for inst in plan:
        tools = tool_recommender.top_k(inst, k=1, thresh=0.3)
        tool_docs.extend([e["doc"] for e in tools])

    tool_docs_str = "\n\n".join(set(tool_docs))
    return tool_docs_str


def write_code(
    coder: LMM,
    chat: List[AgentMessage],
    tool_docs: str,
    plan: str,
) -> str:
    chat = copy.deepcopy(chat)
    if chat[-1].role != "user":
        raise ValueError("Last chat message must be from the user.")

    user_request = chat[-1].content
    prompt = CODE.format(
        docstring=tool_docs,
        question=user_request,
        plan=plan,
    )
    response = cast(str, coder([{"role": "user", "content": prompt}], stream=False))
    maybe_code = extract_tag(response, "code")

    # if the response wasn't properly formatted with the code tags just retrun the response
    if maybe_code is None:
        return response
    return maybe_code


def write_test(
    tester: LMM,
    chat: List[AgentMessage],
    tool_util_docs: str,
    code: str,
    media_list: Optional[Sequence[Union[str, Path]]] = None,
) -> str:
    chat = copy.deepcopy(chat)
    if chat[-1].role != "user":
        raise ValueError("Last chat message must be from the user.")

    user_request = chat[-1].content
    prompt = TEST.format(
        docstring=tool_util_docs,
        question=user_request,
        code=code,
        media=media_list,
    )
    response = cast(str, tester([{"role": "user", "content": prompt}], stream=False))
    maybe_code = extract_tag(response, "code")

    # if the response wasn't properly formatted with the code tags just retrun the response
    if maybe_code is None:
        return response
    return maybe_code


def debug_code(
    debugger: LMM,
    tool_docs: str,
    plan: str,
    code: str,
    test: str,
    result: Execution,
    debug_info: str,
    verbose: bool,
) -> tuple[str, str, str]:
    fixed_code = None
    fixed_test = None
    thoughts = ""
    success = False
    count = 0
    while not success and count < 3:
        try:
            # LLMs write worse code when it's in JSON, so we have it write JSON
            # followed by code each wrapped in markdown blocks.
            fixed_code_and_test_str = debugger(
                FIX_BUG.format(
                    docstring=tool_docs,
                    plan=plan,
                    code=code,
                    tests=test,
                    # Because of the way we trace function calls the trace information
                    # ends up in the results. We don't want to show this info to the
                    # LLM so we don't include it in the tool_output_str.
                    result="\n".join(
                        result.text(include_results=False).splitlines()[-50:]
                    ),
                    debug=debug_info,
                ),
                stream=False,
            )
            fixed_code_and_test_str = cast(str, fixed_code_and_test_str)
            thoughts_tag = extract_tag(fixed_code_and_test_str, "thoughts")
            thoughts = thoughts_tag if thoughts_tag is not None else ""
            fixed_code = extract_tag(fixed_code_and_test_str, "code")
            fixed_test = extract_tag(fixed_code_and_test_str, "test")

            success = not (fixed_code is None and fixed_test is None)

        except Exception as e:
            _CONSOLE.print(f"[bold red]Error while extracting JSON:[/bold red] {e}")

        count += 1

    old_code = code
    old_test = test

    if fixed_code is not None and fixed_code.strip() != "":
        code = fixed_code
    if fixed_test is not None and fixed_test.strip() != "":
        test = fixed_test

    debug_info_i = format_feedback(
        [
            {
                "code": f"{code}\n{test}",
                "feedback": thoughts,
                "edits": get_diff(f"{old_code}\n{old_test}", f"{code}\n{test}"),
            }
        ]
    )
    debug_info += f"\n{debug_info_i}"

    if verbose:
        _CONSOLE.print(
            f"[bold cyan]Thoughts on attempted fix:[/bold cyan] [green]{thoughts}[/green]"
        )

    return code, test, debug_info


def write_and_test_code(
    coder: LMM,
    tester: LMM,
    debugger: LMM,
    chat: List[AgentMessage],
    plan: str,
    tool_docs: str,
    code_interpreter: CodeInterpreter,
    media_list: List[Union[str, Path]],
    verbose: bool,
) -> CodeContext:
    code = write_code(
        coder=coder,
        chat=chat,
        tool_docs=tool_docs,
        plan=plan,
    )
    code = strip_function_calls(code)
    test = write_test(
        tester=tester,
        chat=chat,
        tool_util_docs=T.UTILITIES_DOCSTRING,
        code=code,
        media_list=media_list,
    )
    if verbose:
        print_code("Code:", code)
        print_code("Test:", test)
    result = code_interpreter.exec_isolation(
        f"{DefaultImports.to_code_string()}\n{code}\n{test}"
    )
    if verbose:
        _CONSOLE.print(
            f"[bold cyan]Code execution result:[/bold cyan] [yellow]{escape(result.text(include_logs=True))}[/yellow]"
        )

    count = 0
    debug_info = ""
    while (not result.success or len(result.logs.stdout) == 0) and count < 3:
        code, test, debug_info = debug_code(
            debugger,
            T.UTILITIES_DOCSTRING + "\n" + tool_docs,
            plan,
            code,
            test,
            result,
            debug_info,
            verbose,
        )
        result = code_interpreter.exec_isolation(
            f"{DefaultImports.to_code_string()}\n{code}\n{test}"
        )
        count += 1
        if verbose:
            print_code("Code and test after attempted fix:", code, test)
            _CONSOLE.print(
                f"[bold cyan]Code execution result after attempted fix:[/bold cyan] [yellow]{escape(result.text(include_logs=True))}[/yellow]"
            )

    return CodeContext(
        code=f"{DefaultImports.to_code_string()}\n{code}",
        test=f"{DefaultImports.to_code_string()}\n{test}",
        success=result.success,
        test_result=result,
    )


class VisionAgentCoderV2(AgentCoder):
    """VisionAgentCoderV2 is an agent that will write vision code for you."""

    def __init__(
        self,
        planner: Optional[AgentPlanner] = None,
        coder: Optional[LMM] = None,
        tester: Optional[LMM] = None,
        debugger: Optional[LMM] = None,
        tool_recommender: Optional[Union[str, Sim]] = None,
        verbose: bool = False,
        code_sandbox_runtime: Optional[str] = None,
        update_callback: Callable[[Dict[str, Any]], None] = lambda _: None,
    ) -> None:
        """Initialize the VisionAgentCoderV2.

        Parameters:
            planner (Optional[AgentPlanner]): The planner agent to use for generating
                vision plans. If None, a default VisionAgentPlannerV2 will be used.
            coder (Optional[LMM]): The language model to use for the coder agent. If
                None, a default AnthropicLMM will be used.
            tester (Optional[LMM]): The language model to use for the tester agent. If
                None, a default AnthropicLMM will be used.
            debugger (Optional[LMM]): The language model to use for the debugger agent.
            tool_recommender (Optional[Union[str, Sim]]): The tool recommender to use.
            verbose (bool): Whether to print out debug information.
            code_sandbox_runtime (Optional[str]): The code sandbox runtime to use, can
                be one of: None, "local" or "e2b". If None, it will read from the
                environment variable CODE_SANDBOX_RUNTIME.
            update_callback (Callable[[Dict[str, Any]], None]): The callback function
                that will send back intermediate conversation messages.
        """

        self.planner = (
            planner
            if planner is not None
            else VisionAgentPlannerV2(verbose=verbose, update_callback=update_callback)
        )
        self.coder = (
            coder
            if coder is not None
            else AnthropicLMM(model_name="claude-3-5-sonnet-20241022", temperature=0.0)
        )
        self.tester = (
            tester
            if tester is not None
            else AnthropicLMM(model_name="claude-3-5-sonnet-20241022", temperature=0.0)
        )
        self.debugger = (
            debugger
            if debugger is not None
            else AnthropicLMM(model_name="claude-3-5-sonnet-20241022", temperature=0.0)
        )
        if tool_recommender is not None:
            if isinstance(tool_recommender, str):
                self.tool_recommender = Sim.load(tool_recommender)
            elif isinstance(tool_recommender, Sim):
                self.tool_recommender = tool_recommender
        else:
            self.tool_recommender = load_cached_sim(T.TOOLS_DF)

        self.verbose = verbose
        self.code_sandbox_runtime = code_sandbox_runtime
        self.update_callback = update_callback

    def __call__(
        self,
        input: Union[str, List[Message]],
        media: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate vision code from a conversation.

        Parameters:
            input (Union[str, List[Message]]): The input to the agent. This can be a
                string or a list of messages in the format of [{"role": "user",
                "content": "describe your task here..."}, ...].
            media (Optional[Union[str, Path]]): The path to the media file to use with
                the input. This can be an image or video file.

        Returns:
            str: The generated code as a string.
        """

        input_msg = convert_message_to_agentmessage(input, media)
        return self.generate_code(input_msg).code

    def generate_code(
        self,
        chat: List[AgentMessage],
        max_steps: Optional[int] = None,
        code_interpreter: Optional[CodeInterpreter] = None,
    ) -> CodeContext:
        """Generate vision code from a conversation.

        Parameters:
            chat (List[AgentMessage]): The input to the agent. This should be a list of
                AgentMessage objects.
            code_interpreter (Optional[CodeInterpreter]): The code interpreter to use.

        Returns:
            CodeContext: The generated code as a CodeContext object which includes the
                code, test code, whether or not it was exceuted successfully, and the
                execution result.
        """

        chat = copy.deepcopy(chat)
        with (
            CodeInterpreterFactory.new_instance(self.code_sandbox_runtime)
            if code_interpreter is None
            else code_interpreter
        ) as code_interpreter:
            int_chat, orig_chat, _ = add_media_to_chat(chat, code_interpreter)
            plan_context = self.planner.generate_plan(
                int_chat, max_steps=max_steps, code_interpreter=code_interpreter
            )
            code_context = self.generate_code_from_plan(
                orig_chat,
                plan_context,
                code_interpreter,
            )
        return code_context

    def generate_code_from_plan(
        self,
        chat: List[AgentMessage],
        plan_context: PlanContext,
        code_interpreter: Optional[CodeInterpreter] = None,
    ) -> CodeContext:
        """Generate vision code from a conversation and a previously made plan. This
        will skip the planning step and go straight to generating code.

        Parameters:
            chat (List[AgentMessage]): The input to the agent. This should be a list of
                AgentMessage objects.
            plan_context (PlanContext): The plan context that was previously generated.
            code_interpreter (Optional[CodeInterpreter]): The code interpreter to use.

        Returns:
            CodeContext: The generated code as a CodeContext object which includes the
                code, test code, whether or not it was exceuted successfully, and the
                execution result.
        """

        chat = copy.deepcopy(chat)
        with (
            CodeInterpreterFactory.new_instance(self.code_sandbox_runtime)
            if code_interpreter is None
            else code_interpreter
        ) as code_interpreter:
            int_chat, _, media_list = add_media_to_chat(chat, code_interpreter)
            tool_docs = retrieve_tools(plan_context.instructions, self.tool_recommender)
            code_context = write_and_test_code(
                coder=self.coder,
                tester=self.tester,
                debugger=self.debugger,
                chat=int_chat,
                plan=format_plan_v2(plan_context),
                tool_docs=tool_docs,
                code_interpreter=code_interpreter,
                media_list=media_list,
                verbose=self.verbose,
            )

        self.update_callback(
            {
                "role": "coder",
                "content": format_code_context(code_context),
                "media": capture_media_from_exec(code_context.test_result),
            }
        )
        self.update_callback(
            {
                "role": "observation",
                "content": code_context.test_result.text(),
            }
        )
        return code_context

    def log_progress(self, data: Dict[str, Any]) -> None:
        pass
