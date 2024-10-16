import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast

import libcst as cst
from tabulate import tabulate

import vision_agent.tools as T
from vision_agent.agent.agent import Agent
from vision_agent.agent.agent_utils import (
    _MAX_TABULATE_COL_WIDTH,
    DefaultImports,
    extract_code,
    extract_tag,
    format_memory,
    print_code,
    remove_installs_from_code,
)
from vision_agent.agent.vision_agent_coder_prompts import (
    CODE,
    FIX_BUG,
    FULL_TASK,
    SIMPLE_TEST,
)
from vision_agent.agent.vision_agent_planner import (
    AnthropicVisionAgentPlanner,
    AzureVisionAgentPlanner,
    OllamaVisionAgentPlanner,
    OpenAIVisionAgentPlanner,
    PlanContext,
)
from vision_agent.lmm import (
    LMM,
    AnthropicLMM,
    AzureOpenAILMM,
    Message,
    OllamaLMM,
    OpenAILMM,
)
from vision_agent.tools.meta_tools import get_diff
from vision_agent.utils import CodeInterpreterFactory, Execution
from vision_agent.utils.execute import CodeInterpreter

logging.basicConfig(stream=sys.stdout)
WORKSPACE = Path(os.getenv("WORKSPACE", ""))
_LOGGER = logging.getLogger(__name__)


def strip_function_calls(  # noqa: C901
    code: str, exclusions: Optional[List[str]] = None
) -> str:
    """This will strip out all code that calls functions except for functions included
    in exclusions.
    """
    if exclusions is None:
        exclusions = []

    def check_and_remove_node(node: cst.CSTNode, exclusions: List[str]) -> cst.CSTNode:
        if hasattr(node, "value") and isinstance(node.value, cst.Call):
            if (
                isinstance(node.value.func, cst.Name)
                and node.value.func.value in exclusions
            ):
                return node
            return cst.RemoveFromParent()  # type: ignore
        return node

    class StripFunctionCallsTransformer(cst.CSTTransformer):
        def __init__(self, exclusions: List[str]):
            # Store exclusions to skip removing certain function calls
            self.exclusions = exclusions
            self.in_function_or_class = False

        def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
            self.in_function_or_class = True
            return True

        def leave_FunctionDef(
            self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
        ) -> cst.BaseStatement:
            self.in_function_or_class = False
            return updated_node

        def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
            self.in_function_or_class = True
            return True

        def leave_ClassDef(
            self, node: cst.ClassDef, updated_node: cst.ClassDef
        ) -> cst.BaseStatement:
            self.in_function_or_class = False
            return updated_node

        def leave_Expr(
            self, original_node: cst.Expr, updated_node: cst.Expr
        ) -> cst.Expr:
            if not self.in_function_or_class:
                return cast(
                    cst.Expr, check_and_remove_node(updated_node, self.exclusions)
                )
            return updated_node

        def leave_Assign(
            self, original_node: cst.Assign, updated_node: cst.Assign
        ) -> cst.Assign:
            if not self.in_function_or_class:
                return cast(
                    cst.Assign, check_and_remove_node(updated_node, self.exclusions)
                )
            return updated_node

        def leave_If(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
            if not self.in_function_or_class:
                return cast(
                    cst.If, check_and_remove_node(updated_node, self.exclusions)
                )
            return updated_node

        def leave_For(self, original_node: cst.For, updated_node: cst.For) -> cst.For:
            if not self.in_function_or_class:
                return cast(
                    cst.For, check_and_remove_node(updated_node, self.exclusions)
                )
            return updated_node

        def leave_While(
            self, original_node: cst.While, updated_node: cst.While
        ) -> cst.While:
            if not self.in_function_or_class:
                return cast(
                    cst.While, check_and_remove_node(updated_node, self.exclusions)
                )
            return updated_node

        def leave_With(
            self, original_node: cst.With, updated_node: cst.With
        ) -> cst.With:
            if not self.in_function_or_class:
                return cast(
                    cst.With, check_and_remove_node(updated_node, self.exclusions)
                )
            return updated_node

        def leave_Try(self, original_node: cst.Try, updated_node: cst.Try) -> cst.Try:
            if not self.in_function_or_class:
                return cast(
                    cst.Try, check_and_remove_node(updated_node, self.exclusions)
                )
            return updated_node

    tree = cst.parse_module(code)
    transformer = StripFunctionCallsTransformer(exclusions)
    modified_tree = tree.visit(transformer)
    return modified_tree.code


def write_code(
    coder: LMM,
    chat: List[Message],
    plan: str,
    tool_info: str,
    plan_thoughts: str,
    tool_output: str,
    feedback: str,
) -> str:
    chat = copy.deepcopy(chat)
    if chat[-1]["role"] != "user":
        raise ValueError("Last chat message must be from the user.")

    user_request = chat[-1]["content"]
    prompt = CODE.format(
        docstring=tool_info,
        question=FULL_TASK.format(user_request=user_request, subtasks=plan),
        tool_output=tool_output,
        plan_thoughts=plan_thoughts,
        feedback=feedback,
    )
    chat[-1]["content"] = prompt
    return extract_code(coder(chat, stream=False))  # type: ignore


def write_test(
    tester: LMM,
    chat: List[Message],
    tool_utils: str,
    code: str,
    feedback: str,
    media: Optional[Sequence[Union[str, Path]]] = None,
) -> str:
    chat = copy.deepcopy(chat)
    if chat[-1]["role"] != "user":
        raise ValueError("Last chat message must be from the user.")

    user_request = chat[-1]["content"]
    prompt = SIMPLE_TEST.format(
        docstring=tool_utils,
        question=user_request,
        code=code,
        feedback=feedback,
        media=media,
    )
    chat[-1]["content"] = prompt
    return extract_code(tester(chat, stream=False))  # type: ignore


def write_and_test_code(
    chat: List[Message],
    plan: str,
    tool_info: str,
    tool_output: str,
    plan_thoughts: str,
    tool_utils: str,
    working_memory: List[Dict[str, str]],
    coder: LMM,
    tester: LMM,
    debugger: LMM,
    code_interpreter: CodeInterpreter,
    log_progress: Callable[[Dict[str, Any]], None],
    verbosity: int = 0,
    max_retries: int = 3,
    media: Optional[Sequence[Union[str, Path]]] = None,
) -> Dict[str, Any]:
    log_progress(
        {
            "type": "log",
            "log_content": "Generating code",
            "status": "started",
        }
    )
    code = write_code(
        coder,
        chat,
        plan,
        tool_info,
        tool_output,
        plan_thoughts,
        format_memory(working_memory),
    )
    code = strip_function_calls(code)
    test = write_test(
        tester, chat, tool_utils, code, format_memory(working_memory), media
    )

    log_progress(
        {
            "type": "log",
            "log_content": "Running code",
            "status": "running",
            "code": DefaultImports.prepend_imports(code),
            "payload": {
                "test": test,
            },
        }
    )
    result = code_interpreter.exec_isolation(
        f"{DefaultImports.to_code_string()}\n{code}\n{test}"
    )
    log_progress(
        {
            "type": "log",
            "log_content": (
                "Code execution succeeded"
                if result.success
                else "Code execution failed"
            ),
            "status": "completed" if result.success else "failed",
            "code": DefaultImports.prepend_imports(code),
            "payload": {
                "test": test,
            },
        }
    )
    if verbosity == 2:
        print_code("Initial code and tests:", code, test)
        _LOGGER.info(
            f"Initial code execution result:\n{result.text(include_logs=True)}"
        )

    count = 0
    new_working_memory: List[Dict[str, str]] = []
    while not result.success and count < max_retries:
        if verbosity == 2:
            _LOGGER.info(f"Start debugging attempt {count + 1}")
        code, test, result = debug_code(
            working_memory,
            debugger,
            code_interpreter,
            tool_info,
            code,
            test,
            result,
            new_working_memory,
            log_progress,
            verbosity,
        )
        count += 1

    if verbosity >= 1:
        print_code("Final code and tests:", code, test)

    return {
        "code": code,
        "test": test,
        "success": result.success,
        "test_result": result,
        "working_memory": new_working_memory,
    }


def debug_code(
    working_memory: List[Dict[str, str]],
    debugger: LMM,
    code_interpreter: CodeInterpreter,
    tool_info: str,
    code: str,
    test: str,
    result: Execution,
    new_working_memory: List[Dict[str, str]],
    log_progress: Callable[[Dict[str, Any]], None],
    verbosity: int = 0,
) -> tuple[str, str, Execution]:
    log_progress(
        {
            "type": "log",
            "log_content": ("Debugging code"),
            "status": "started",
        }
    )

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
                    docstring=tool_info,
                    code=code,
                    tests=test,
                    # Because of the way we trace function calls the trace information
                    # ends up in the results. We don't want to show this info to the
                    # LLM so we don't include it in the tool_output_str.
                    result="\n".join(
                        result.text(include_results=False).splitlines()[-50:]
                    ),
                    feedback=format_memory(working_memory + new_working_memory),
                ),
                stream=False,
            )
            fixed_code_and_test_str = cast(str, fixed_code_and_test_str)
            thoughts_tag = extract_tag(fixed_code_and_test_str, "thoughts")
            thoughts = thoughts_tag if thoughts_tag is not None else ""
            fixed_code = extract_tag(fixed_code_and_test_str, "code")
            fixed_test = extract_tag(fixed_code_and_test_str, "test")

            if fixed_code is None and fixed_test is None:
                success = False
            else:
                success = True

        except Exception as e:
            _LOGGER.exception(f"Error while extracting JSON: {e}")

        count += 1

    old_code = code
    old_test = test

    if fixed_code is not None and fixed_code.strip() != "":
        code = fixed_code
    if fixed_test is not None and fixed_test.strip() != "":
        test = fixed_test

    new_working_memory.append(
        {
            "code": f"{code}\n{test}",
            "feedback": thoughts,
            "edits": get_diff(f"{old_code}\n{old_test}", f"{code}\n{test}"),
        }
    )
    log_progress(
        {
            "type": "log",
            "log_content": ("Running code"),
            "status": "running",
            "code": DefaultImports.prepend_imports(code),
            "payload": {
                "test": test,
            },
        }
    )

    result = code_interpreter.exec_isolation(
        f"{DefaultImports.to_code_string()}\n{code}\n{test}"
    )
    log_progress(
        {
            "type": "log",
            "log_content": (
                "Code execution succeed" if result.success else "Code execution failed"
            ),
            "status": "completed" if result.success else "failed",
            "code": DefaultImports.prepend_imports(code),
            "payload": {
                "test": test,
                # "result": result.to_json(),
            },
        }
    )
    if verbosity == 2:
        print_code("Code and test after attempted fix:", code, test)
        _LOGGER.info(
            f"Reflection: {thoughts}\nCode execution result after attempted fix: {result.text(include_logs=True)}"
        )

    return code, test, result


class VisionAgentCoder(Agent):
    """Vision Agent Coder is an agentic framework that can output code based on a user
    request. It can plan tasks, retrieve relevant tools, write code, write tests and
    reflect on failed test cases to debug code. It is inspired by AgentCoder
    https://arxiv.org/abs/2312.13010 and Data Interpeter https://arxiv.org/abs/2402.18679

    Example
    -------
        >>> import vision_agent as va
        >>> agent = va.agent.VisionAgentCoder()
        >>> code = agent("What percentage of the area of the jar is filled with coffee beans?", media="jar.jpg")
    """

    def __init__(
        self,
        planner: Optional[Agent] = None,
        coder: Optional[LMM] = None,
        tester: Optional[LMM] = None,
        debugger: Optional[LMM] = None,
        verbosity: int = 0,
        report_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        code_interpreter: Optional[Union[str, CodeInterpreter]] = None,
    ) -> None:
        """Initialize the Vision Agent Coder.

        Parameters:
            planner (Optional[Agent]): The planner model to use. Defaults to
                AnthropicVisionAgentPlanner.
            coder (Optional[LMM]): The coder model to use. Defaults to AnthropicLMM.
            tester (Optional[LMM]): The tester model to use. Defaults to AnthropicLMM.
            debugger (Optional[LMM]): The debugger model to use. Defaults to AnthropicLMM.
            verbosity (int): The verbosity level of the agent. Defaults to 0. 2 is the
                highest verbosity level which will output all intermediate debugging
                code.
            report_progress_callback (Optional[Callable[Dict[str, Any]]]): a callback
                to report the progress of the agent. This is useful for streaming logs
                in a web application where multiple VisionAgentCoder instances are
                running in parallel. This callback ensures that the progress are not
                mixed up.
            code_interpreter (Optional[Union[str, CodeInterpreter]]): For string values
                it can be one of: None, "local" or "e2b". If None, it will read from
                the environment variable "CODE_SANDBOX_RUNTIME". If a CodeInterpreter
                object is provided it will use that.
        """

        self.planner = (
            AnthropicVisionAgentPlanner(verbosity=verbosity)
            if planner is None
            else planner
        )
        self.coder = AnthropicLMM(temperature=0.0) if coder is None else coder
        self.tester = AnthropicLMM(temperature=0.0) if tester is None else tester
        self.debugger = AnthropicLMM(temperature=0.0) if debugger is None else debugger
        self.verbosity = verbosity
        if self.verbosity > 0:
            _LOGGER.setLevel(logging.INFO)

        self.report_progress_callback = report_progress_callback
        self.code_interpreter = code_interpreter

    def __call__(
        self,
        input: Union[str, List[Message]],
        media: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate code based on a user request.

        Parameters:
            input (Union[str, List[Message]]): A conversation in the format of
                [{"role": "user", "content": "describe your task here..."}] or a string
                of just the contents.
            media (Optional[Union[str, Path]]): The media file to be used in the task.

        Returns:
            str: The code output by the VisionAgentCoder.
        """

        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
            if media is not None:
                input[0]["media"] = [media]
        code_and_context = self.generate_code(input)
        return code_and_context["code"]  # type: ignore

    def generate_code_from_plan(
        self,
        chat: List[Message],
        plan_context: PlanContext,
        code_interpreter: Optional[CodeInterpreter] = None,
    ) -> Dict[str, Any]:
        """Generates code and other intermediate outputs from a chat input and a plan.
        The plan includes:
            - plans: The plans generated by the planner.
            - best_plan: The best plan selected by the planner.
            - plan_thoughts: The thoughts of the planner, including any modifications
                to the plan.
            - tool_doc: The tool documentation for the best plan.
            - tool_output: The tool output from the tools used by the best plan.

        Parameters:
            chat (List[Message]): A conversation in the format of
                [{"role": "user", "content": "describe your task here..."}].
            plan_context (PlanContext): The context of the plan, including the plans,
                best_plan, plan_thoughts, tool_doc, and tool_output.
            test_multi_plan (bool): Whether to test multiple plans or just the best plan.
            custom_tool_names (Optional[List[str]]): A list of custom tool names to use
                for the planner.

        Returns:
            Dict[str, Any]: A dictionary containing the code output by the
                VisionAgentCoder and other intermediate outputs. include:
                - status (str): Whether or not the agent completed or failed generating
                    the code.
                - code (str): The code output by the VisionAgentCoder.
                - test (str): The test output by the VisionAgentCoder.
                - test_result (Execution): The result of the test execution.
                - plans (Dict[str, Any]): The plans generated by the planner.
                - plan_thoughts (str): The thoughts of the planner.
                - working_memory (List[Dict[str, str]]): The working memory of the agent.
        """
        if not chat:
            raise ValueError("Chat cannot be empty.")

        # NOTE: each chat should have a dedicated code interpreter instance to avoid concurrency issues
        code_interpreter = (
            self.code_interpreter
            if self.code_interpreter is not None
            and not isinstance(self.code_interpreter, str)
            else CodeInterpreterFactory.new_instance(
                code_sandbox_runtime=self.code_interpreter,
            )
        )
        with code_interpreter:
            chat = copy.deepcopy(chat)
            media_list = []
            for chat_i in chat:
                if "media" in chat_i:
                    for media in chat_i["media"]:
                        media = (
                            media
                            if type(media) is str
                            and media.startswith(("http", "https"))
                            else code_interpreter.upload_file(cast(str, media))
                        )
                        chat_i["content"] += f" Media name {media}"  # type: ignore
                        media_list.append(str(media))

            int_chat = cast(
                List[Message],
                [
                    (
                        {
                            "role": c["role"],
                            "content": c["content"],
                            "media": c["media"],
                        }
                        if "media" in c
                        else {"role": c["role"], "content": c["content"]}
                    )
                    for c in chat
                ],
            )

            code = ""
            test = ""
            working_memory: List[Dict[str, str]] = []
            plan = plan_context.plans[plan_context.best_plan]
            tool_doc = plan_context.tool_doc
            tool_output_str = plan_context.tool_output
            plan_thoughts_str = str(plan_context.plan_thoughts)

            if self.verbosity >= 1:
                plan_fixed = [{"instructions": e} for e in plan["instructions"]]
                _LOGGER.info(
                    f"Picked best plan:\n{tabulate(tabular_data=plan_fixed, headers='keys', tablefmt='mixed_grid', maxcolwidths=_MAX_TABULATE_COL_WIDTH)}"
                )

            results = write_and_test_code(
                chat=[{"role": c["role"], "content": c["content"]} for c in int_chat],
                plan=f"\n{plan['thoughts']}\n-"
                + "\n-".join([e for e in plan["instructions"]]),
                tool_info=tool_doc,
                tool_output=tool_output_str,
                plan_thoughts=plan_thoughts_str,
                tool_utils=T.UTILITIES_DOCSTRING,
                working_memory=working_memory,
                coder=self.coder,
                tester=self.tester,
                debugger=self.debugger,
                code_interpreter=code_interpreter,
                log_progress=self.log_progress,
                verbosity=self.verbosity,
                media=media_list,
            )
            success = cast(bool, results["success"])
            code = remove_installs_from_code(cast(str, results["code"]))
            test = remove_installs_from_code(cast(str, results["test"]))
            working_memory.extend(results["working_memory"])
            execution_result = cast(Execution, results["test_result"])

            return {
                "status": "completed" if success else "failed",
                "code": DefaultImports.prepend_imports(code),
                "test": test,
                "test_result": execution_result,
                "plans": plan_context.plans,
                "plan_thoughts": plan_thoughts_str,
                "working_memory": working_memory,
            }

    def generate_code(
        self,
        chat: List[Message],
        test_multi_plan: bool = True,
        custom_tool_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generates code and other intermediate outputs from a chat input.

        Parameters:
            chat (List[Message]): A conversation in the format of
                [{"role": "user", "content": "describe your task here..."}].
            test_multi_plan (bool): Whether to test multiple plans or just the best plan.
            custom_tool_names (Optional[List[str]]): A list of custom tool names to use
                for the planner.

        Returns:
            Dict[str, Any]: A dictionary containing the code output by the
                VisionAgentCoder and other intermediate outputs. include:
                - status (str): Whether or not the agent completed or failed generating
                    the code.
                - code (str): The code output by the VisionAgentCoder.
                - test (str): The test output by the VisionAgentCoder.
                - test_result (Execution): The result of the test execution.
                - plans (Dict[str, Any]): The plans generated by the planner.
                - plan_thoughts (str): The thoughts of the planner.
                - working_memory (List[Dict[str, str]]): The working memory of the agent.
        """
        if not chat:
            raise ValueError("Chat cannot be empty.")

        # NOTE: each chat should have a dedicated code interpreter instance to avoid concurrency issues
        code_interpreter = (
            self.code_interpreter
            if self.code_interpreter is not None
            and not isinstance(self.code_interpreter, str)
            else CodeInterpreterFactory.new_instance(
                code_sandbox_runtime=self.code_interpreter,
            )
        )
        with code_interpreter:
            plan_context = self.planner.generate_plan(  # type: ignore
                chat,
                test_multi_plan=test_multi_plan,
                custom_tool_names=custom_tool_names,
                code_interpreter=code_interpreter,
            )

            code_and_context = self.generate_code_from_plan(
                chat,
                plan_context,
                code_interpreter=code_interpreter,
            )
        return code_and_context

    def chat(self, chat: List[Message]) -> List[Message]:
        chat = copy.deepcopy(chat)
        code = self.generate_code(chat)
        chat.append({"role": "agent", "content": code["code"]})
        return chat

    def log_progress(self, data: Dict[str, Any]) -> None:
        if self.report_progress_callback is not None:
            self.report_progress_callback(data)


class OpenAIVisionAgentCoder(VisionAgentCoder):
    """Initializes Vision Agent Coder using OpenAI models for planning, coding, testing."""

    def __init__(
        self,
        planner: Optional[Agent] = None,
        coder: Optional[LMM] = None,
        tester: Optional[LMM] = None,
        debugger: Optional[LMM] = None,
        verbosity: int = 0,
        report_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        code_interpreter: Optional[Union[str, CodeInterpreter]] = None,
    ) -> None:
        self.planner = (
            OpenAIVisionAgentPlanner(verbosity=verbosity)
            if planner is None
            else planner
        )
        self.coder = OpenAILMM(temperature=0.0) if coder is None else coder
        self.tester = OpenAILMM(temperature=0.0) if tester is None else tester
        self.debugger = OpenAILMM(temperature=0.0) if debugger is None else debugger
        self.verbosity = verbosity
        if self.verbosity > 0:
            _LOGGER.setLevel(logging.INFO)

        self.report_progress_callback = report_progress_callback
        self.code_interpreter = code_interpreter


class AnthropicVisionAgentCoder(VisionAgentCoder):
    """Initializes Vision Agent Coder using Anthropic models for planning, coding, testing."""

    def __init__(
        self,
        planner: Optional[Agent] = None,
        coder: Optional[LMM] = None,
        tester: Optional[LMM] = None,
        debugger: Optional[LMM] = None,
        verbosity: int = 0,
        report_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        code_interpreter: Optional[Union[str, CodeInterpreter]] = None,
    ) -> None:
        # NOTE: Claude doesn't have an official JSON mode
        self.planner = (
            AnthropicVisionAgentPlanner(verbosity=verbosity)
            if planner is None
            else planner
        )
        self.coder = AnthropicLMM(temperature=0.0) if coder is None else coder
        self.tester = AnthropicLMM(temperature=0.0) if tester is None else tester
        self.debugger = AnthropicLMM(temperature=0.0) if debugger is None else debugger
        self.verbosity = verbosity
        if self.verbosity > 0:
            _LOGGER.setLevel(logging.INFO)

        self.report_progress_callback = report_progress_callback
        self.code_interpreter = code_interpreter


class OllamaVisionAgentCoder(VisionAgentCoder):
    """VisionAgentCoder that uses Ollama models for planning, coding, testing.

    Pre-requisites:
    1. Run ollama pull llama3.1 for the LLM
    2. Run ollama pull mxbai-embed-large for the embedding similarity model

    Technically you should use a VLM such as llava but llava is not able to handle the
    context length and crashes.

    Example
    -------
        >>> image vision_agent as va
        >>> agent = va.agent.OllamaVisionAgentCoder()
        >>> code = agent("What percentage of the area of the jar is filled with coffee beans?", media="jar.jpg")
    """

    def __init__(
        self,
        planner: Optional[Agent] = None,
        coder: Optional[LMM] = None,
        tester: Optional[LMM] = None,
        debugger: Optional[LMM] = None,
        verbosity: int = 0,
        report_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        code_interpreter: Optional[Union[str, CodeInterpreter]] = None,
    ) -> None:
        super().__init__(
            planner=(
                OllamaVisionAgentPlanner(verbosity=verbosity)
                if planner is None
                else planner
            ),
            coder=(
                OllamaLMM(model_name="llama3.1", temperature=0.0)
                if coder is None
                else coder
            ),
            tester=(
                OllamaLMM(model_name="llama3.1", temperature=0.0)
                if tester is None
                else tester
            ),
            debugger=(
                OllamaLMM(model_name="llama3.1", temperature=0.0)
                if debugger is None
                else debugger
            ),
            verbosity=verbosity,
            report_progress_callback=report_progress_callback,
            code_interpreter=code_interpreter,
        )


class AzureVisionAgentCoder(VisionAgentCoder):
    """VisionAgentCoder that uses Azure OpenAI APIs for planning, coding, testing.

    Pre-requisites:
    1. Set the environment variable AZURE_OPENAI_API_KEY to your Azure OpenAI API key.
    2. Set the environment variable AZURE_OPENAI_ENDPOINT to your Azure OpenAI endpoint.

    Example
    -------
        >>> import vision_agent as va
        >>> agent = va.agent.AzureVisionAgentCoder()
        >>> code = agent("What percentage of the area of the jar is filled with coffee beans?", media="jar.jpg")
    """

    def __init__(
        self,
        planner: Optional[Agent] = None,
        coder: Optional[LMM] = None,
        tester: Optional[LMM] = None,
        debugger: Optional[LMM] = None,
        verbosity: int = 0,
        report_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        code_interpreter: Optional[Union[str, CodeInterpreter]] = None,
    ) -> None:
        """Initialize the Vision Agent Coder.

        Parameters:
            planner (Optional[Agent]): The planner model to use. Defaults to
                AzureVisionAgentPlanner.
            coder (Optional[LMM]): The coder model to use. Defaults to OpenAILMM.
            tester (Optional[LMM]): The tester model to use. Defaults to OpenAILMM.
            debugger (Optional[LMM]): The debugger model to
            verbosity (int): The verbosity level of the agent. Defaults to 0. 2 is the
                highest verbosity level which will output all intermediate debugging
                code.
            report_progress_callback: a callback to report the progress of the agent.
                This is useful for streaming logs in a web application where multiple
                VisionAgentCoder instances are running in parallel. This callback
                ensures that the progress are not mixed up.
        """
        super().__init__(
            planner=(
                AzureVisionAgentPlanner(verbosity=verbosity)
                if planner is None
                else planner
            ),
            coder=AzureOpenAILMM(temperature=0.0) if coder is None else coder,
            tester=AzureOpenAILMM(temperature=0.0) if tester is None else tester,
            debugger=(
                AzureOpenAILMM(temperature=0.0) if debugger is None else debugger
            ),
            verbosity=verbosity,
            report_progress_callback=report_progress_callback,
            code_interpreter=code_interpreter,
        )
