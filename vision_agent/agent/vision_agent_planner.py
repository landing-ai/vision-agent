import copy
import logging
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from pydantic import BaseModel
from tabulate import tabulate

import vision_agent.tools as T
from vision_agent.agent import Agent
from vision_agent.agent.vision_agent_planner_prompts import (
    PICK_PLAN,
    PLAN,
    PREVIOUS_FAILED,
    TEST_PLANS,
    USER_REQ,
)
from vision_agent.lmm import LMM, AnthropicLMM, AzureOpenAILMM, OllamaLMM, OpenAILMM
from vision_agent.models import Message
from vision_agent.sim import AzureSim, OllamaSim, Sim
from vision_agent.utils.agent import (
    _MAX_TABULATE_COL_WIDTH,
    DefaultImports,
    extract_code,
    extract_json,
    format_feedback,
    format_plans,
    print_code,
)
from vision_agent.utils.execute import (
    CodeInterpreter,
    CodeInterpreterFactory,
    Execution,
)
from vision_agent.utils.tools_doc import get_tool_descriptions_by_names

_LOGGER = logging.getLogger(__name__)


class PlanContext(BaseModel):
    plans: Dict[str, Dict[str, Union[str, List[str]]]]
    best_plan: str
    plan_thoughts: str
    tool_output: str
    tool_doc: str
    test_results: Optional[Execution]


def retrieve_tools(
    plans: Dict[str, Dict[str, Any]],
    tool_recommender: Sim,
    log_progress: Callable[[Dict[str, Any]], None],
    verbosity: int = 0,
) -> Dict[str, str]:
    log_progress(
        {
            "type": "log",
            "log_content": ("Retrieving tools for each plan"),
            "status": "started",
        }
    )
    tool_info = []
    tool_desc = []
    tool_lists: Dict[str, List[Dict[str, str]]] = {}
    for k, plan in plans.items():
        tool_lists[k] = []
        for task in plan["instructions"]:
            tools = tool_recommender.top_k(task, k=2, thresh=0.3)
            tool_info.extend([e["doc"] for e in tools])
            tool_desc.extend([e["desc"] for e in tools])
            tool_lists[k].extend(
                {"description": e["desc"], "documentation": e["doc"]} for e in tools
            )

    if verbosity == 2:
        tool_desc_str = "\n".join(set(tool_desc))
        _LOGGER.info(f"Tools Description:\n{tool_desc_str}")

    tool_lists_unique = {}
    for k in tool_lists:
        tool_lists_unique[k] = "\n\n".join(
            set(e["documentation"] for e in tool_lists[k])
        )
    all_tools = "\n\n".join(set(tool_info))
    tool_lists_unique["all"] = all_tools
    return tool_lists_unique


def _check_plan_format(plan: Dict[str, Any]) -> bool:
    if not isinstance(plan, dict):
        return False

    for k in plan:
        if "thoughts" not in plan[k] or "instructions" not in plan[k]:
            return False
        if not isinstance(plan[k]["instructions"], list):
            return False
    return True


def write_plans(
    chat: List[Message], tool_desc: str, working_memory: str, model: LMM
) -> Dict[str, Any]:
    chat = copy.deepcopy(chat)
    if chat[-1]["role"] != "user":
        raise ValueError("Last message in chat must be from user")

    user_request = chat[-1]["content"]
    context = USER_REQ.format(user_request=user_request)
    prompt = PLAN.format(
        context=context,
        tool_desc=tool_desc,
        feedback=working_memory,
    )
    chat[-1]["content"] = prompt
    plans = extract_json(model(chat, stream=False))  # type: ignore

    count = 0
    while not _check_plan_format(plans) and count < 3:
        _LOGGER.info("Invalid plan format. Retrying.")
        plans = extract_json(model(chat, stream=False))  # type: ignore
        count += 1
        if count == 3:
            raise ValueError("Failed to generate valid plans after 3 attempts.")
    return plans


def write_and_exec_plan_tests(
    plans: Dict[str, Any],
    tool_info: str,
    media: List[str],
    model: LMM,
    log_progress: Callable[[Dict[str, Any]], None],
    code_interpreter: CodeInterpreter,
    verbosity: int = 0,
    max_retries: int = 3,
) -> Tuple[str, Execution]:

    plan_str = format_plans(plans)
    prompt = TEST_PLANS.format(
        docstring=tool_info, plans=plan_str, previous_attempts="", media=media
    )

    code = extract_code(model(prompt, stream=False))  # type: ignore
    log_progress(
        {
            "type": "log",
            "log_content": "Executing code to test plans",
            "code": DefaultImports.prepend_imports(code),
            "status": "running",
        }
    )
    tool_output = code_interpreter.exec_isolation(DefaultImports.prepend_imports(code))
    # Because of the way we trace function calls the trace information ends up in the
    # results. We don't want to show this info to the LLM so we don't include it in the
    # tool_output_str.
    tool_output_str = tool_output.text(include_results=False).strip()

    if verbosity == 2:
        print_code("Initial code and tests:", code)
        _LOGGER.info(f"Initial code execution result:\n{tool_output_str}")

    log_progress(
        {
            "type": "log",
            "log_content": (
                "Code execution succeeded"
                if tool_output.success
                else "Code execution failed"
            ),
            "code": DefaultImports.prepend_imports(code),
            # "payload": tool_output.to_json(),
            "status": "completed" if tool_output.success else "failed",
        }
    )

    # retry if the tool output is empty or code fails
    count = 0
    tool_output_str = tool_output.text(include_results=False).strip()
    while (
        not tool_output.success
        or (len(tool_output.logs.stdout) == 0 and len(tool_output.logs.stderr) == 0)
    ) and count < max_retries:
        prompt = TEST_PLANS.format(
            docstring=tool_info,
            plans=plan_str,
            previous_attempts=PREVIOUS_FAILED.format(
                code=code, error="\n".join(tool_output_str.splitlines()[-50:])
            ),
            media=media,
        )
        log_progress(
            {
                "type": "log",
                "log_content": "Retrying code to test plans",
                "status": "running",
                "code": DefaultImports.prepend_imports(code),
            }
        )
        code = extract_code(model(prompt, stream=False))  # type: ignore
        tool_output = code_interpreter.exec_isolation(
            DefaultImports.prepend_imports(code)
        )
        log_progress(
            {
                "type": "log",
                "log_content": (
                    "Code execution succeeded"
                    if tool_output.success
                    else "Code execution failed"
                ),
                "code": DefaultImports.prepend_imports(code),
                # "payload": tool_output.to_json(),
                "status": "completed" if tool_output.success else "failed",
            }
        )
        tool_output_str = tool_output.text(include_results=False).strip()

        if verbosity == 2:
            print_code("Code and test after attempted fix:", code)
            _LOGGER.info(f"Code execution result after attempt {count + 1}")
            _LOGGER.info(f"{tool_output_str}")

        count += 1

    return code, tool_output


def write_plan_thoughts(
    chat: List[Message],
    plans: Dict[str, Any],
    tool_output_str: str,
    model: LMM,
    max_retries: int = 3,
) -> Dict[str, str]:
    user_req = chat[-1]["content"]
    context = USER_REQ.format(user_request=user_req)
    # because the tool picker model gets the image as well, we have to be careful with
    # how much text we send it, so we truncate the tool output to 20,000 characters
    prompt = PICK_PLAN.format(
        context=context,
        plans=format_plans(plans),
        tool_output=tool_output_str[:20_000],
    )
    chat[-1]["content"] = prompt
    count = 0

    plan_thoughts = None
    while plan_thoughts is None and count < max_retries:
        try:
            plan_thoughts = extract_json(model(chat, stream=False))  # type: ignore
        except JSONDecodeError as e:
            _LOGGER.exception(
                f"Error while extracting JSON during picking best plan {str(e)}"
            )
            pass
        count += 1

    if (
        plan_thoughts is None
        or "best_plan" not in plan_thoughts
        or ("best_plan" in plan_thoughts and plan_thoughts["best_plan"] not in plans)
    ):
        _LOGGER.info(f"Failed to pick best plan. Using the first plan. {plan_thoughts}")
        plan_thoughts = {"best_plan": list(plans.keys())[0]}

    if "thoughts" not in plan_thoughts:
        plan_thoughts["thoughts"] = ""
    return plan_thoughts


def pick_plan(
    chat: List[Message],
    plans: Dict[str, Any],
    tool_info: str,
    model: LMM,
    code_interpreter: CodeInterpreter,
    media: List[str],
    log_progress: Callable[[Dict[str, Any]], None],
    verbosity: int = 0,
    max_retries: int = 3,
) -> Tuple[Dict[str, str], str, Execution]:
    log_progress(
        {
            "type": "log",
            "log_content": "Generating code to pick the best plan",
            "status": "started",
        }
    )

    chat = copy.deepcopy(chat)
    if chat[-1]["role"] != "user":
        raise ValueError("Last chat message must be from the user.")

    code, tool_output = write_and_exec_plan_tests(
        plans,
        tool_info,
        media,
        model,
        log_progress,
        code_interpreter,
        verbosity,
        max_retries,
    )

    if verbosity >= 1:
        print_code("Final code:", code)

    plan_thoughts = write_plan_thoughts(
        chat,
        plans,
        tool_output.text(include_results=False).strip(),
        model,
        max_retries,
    )

    if verbosity >= 1:
        _LOGGER.info(f"Best plan:\n{plan_thoughts}")
    log_progress(
        {
            "type": "log",
            "log_content": "Picked best plan",
            "status": "completed",
            "payload": plans[plan_thoughts["best_plan"]],
        }
    )
    return plan_thoughts, code, tool_output


class VisionAgentPlanner(Agent):
    def __init__(
        self,
        planner: Optional[LMM] = None,
        tool_recommender: Optional[Sim] = None,
        verbosity: int = 0,
        report_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        code_interpreter: Optional[Union[str, CodeInterpreter]] = None,
    ) -> None:
        self.planner = AnthropicLMM(temperature=0.0) if planner is None else planner
        self.verbosity = verbosity
        if self.verbosity > 0:
            _LOGGER.setLevel(logging.INFO)

        self.tool_recommender = (
            Sim(T.get_tools_df(), sim_key="desc")
            if tool_recommender is None
            else tool_recommender
        )
        self.report_progress_callback = report_progress_callback
        self.code_interpreter = code_interpreter

    def __call__(
        self, input: Union[str, List[Message]], media: Optional[Union[str, Path]] = None
    ) -> str:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
            if media is not None:
                input[0]["media"] = [media]
        planning_context = self.generate_plan(input)
        return str(planning_context.plans[planning_context.best_plan])

    def generate_plan(
        self,
        chat: List[Message],
        test_multi_plan: bool = True,
        custom_tool_names: Optional[List[str]] = None,
        code_interpreter: Optional[CodeInterpreter] = None,
    ) -> PlanContext:
        if not chat:
            raise ValueError("Chat cannot be empty")

        code_interpreter = (
            code_interpreter
            if code_interpreter is not None
            else (
                self.code_interpreter
                if not isinstance(self.code_interpreter, str)
                else CodeInterpreterFactory.new_instance(self.code_interpreter)
            )
        )
        code_interpreter = cast(CodeInterpreter, code_interpreter)
        with code_interpreter:
            chat = copy.deepcopy(chat)
            media_list = []
            for chat_i in chat:
                if "media" in chat_i:
                    for media in chat_i["media"]:
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

            working_memory: List[Dict[str, str]] = []

            plans = write_plans(
                chat,
                get_tool_descriptions_by_names(
                    custom_tool_names, T.tools.FUNCTION_TOOLS, T.tools.UTIL_TOOLS  # type: ignore
                ),
                format_feedback(working_memory),
                self.planner,
            )
            if self.verbosity >= 1:
                for plan in plans:
                    plan_fixed = [
                        {"instructions": e} for e in plans[plan]["instructions"]
                    ]
                    _LOGGER.info(
                        f"\n{tabulate(tabular_data=plan_fixed, headers='keys', tablefmt='mixed_grid', maxcolwidths=_MAX_TABULATE_COL_WIDTH)}"
                    )

            tool_docs = retrieve_tools(
                plans,
                self.tool_recommender,
                self.log_progress,
                self.verbosity,
            )
            if test_multi_plan:
                plan_thoughts, code, tool_output = pick_plan(
                    int_chat,
                    plans,
                    tool_docs["all"],
                    self.planner,
                    code_interpreter,
                    media_list,
                    self.log_progress,
                    self.verbosity,
                )
                best_plan = plan_thoughts["best_plan"]
                plan_thoughts_str = plan_thoughts["thoughts"]
                tool_output_str = (
                    "```python\n"
                    + code
                    + "\n```\n"
                    + tool_output.text(include_results=False).strip()
                )
            else:
                best_plan = list(plans.keys())[0]
                tool_output_str = ""
                plan_thoughts_str = ""
                tool_output = None

            if best_plan in plans and best_plan in tool_docs:
                tool_doc = tool_docs[best_plan]
            else:
                if self.verbosity >= 1:
                    _LOGGER.warning(
                        f"Best plan {best_plan} not found in plans or tool_infos. Using the first plan and tool info."
                    )
                k = list(plans.keys())[0]
                best_plan = k
                tool_doc = tool_docs[k]

        return PlanContext(
            plans=plans,
            best_plan=best_plan,
            plan_thoughts=plan_thoughts_str,
            tool_output=tool_output_str,
            test_results=tool_output,
            tool_doc=tool_doc,
        )

    def log_progress(self, log: Dict[str, Any]) -> None:
        if self.report_progress_callback is not None:
            self.report_progress_callback(log)


class AnthropicVisionAgentPlanner(VisionAgentPlanner):
    def __init__(
        self,
        planner: Optional[LMM] = None,
        tool_recommender: Optional[Sim] = None,
        verbosity: int = 0,
        report_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        code_interpreter: Optional[Union[str, CodeInterpreter]] = None,
    ) -> None:
        super().__init__(
            planner=AnthropicLMM(temperature=0.0) if planner is None else planner,
            tool_recommender=tool_recommender,
            verbosity=verbosity,
            report_progress_callback=report_progress_callback,
            code_interpreter=code_interpreter,
        )


class OpenAIVisionAgentPlanner(VisionAgentPlanner):
    def __init__(
        self,
        planner: Optional[LMM] = None,
        tool_recommender: Optional[Sim] = None,
        verbosity: int = 0,
        report_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        code_interpreter: Optional[Union[str, CodeInterpreter]] = None,
    ) -> None:
        super().__init__(
            planner=(OpenAILMM(temperature=0.0) if planner is None else planner),
            tool_recommender=tool_recommender,
            verbosity=verbosity,
            report_progress_callback=report_progress_callback,
            code_interpreter=code_interpreter,
        )


class OllamaVisionAgentPlanner(VisionAgentPlanner):
    def __init__(
        self,
        planner: Optional[LMM] = None,
        tool_recommender: Optional[Sim] = None,
        verbosity: int = 0,
        report_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        code_interpreter: Optional[Union[str, CodeInterpreter]] = None,
    ) -> None:
        super().__init__(
            planner=(
                OllamaLMM(model_name="llama3.2-vision", temperature=0.0)
                if planner is None
                else planner
            ),
            tool_recommender=(
                OllamaSim(T.get_tools_df(), sim_key="desc")
                if tool_recommender is None
                else tool_recommender
            ),
            verbosity=verbosity,
            report_progress_callback=report_progress_callback,
            code_interpreter=code_interpreter,
        )


class AzureVisionAgentPlanner(VisionAgentPlanner):
    def __init__(
        self,
        planner: Optional[LMM] = None,
        tool_recommender: Optional[Sim] = None,
        verbosity: int = 0,
        report_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        code_interpreter: Optional[Union[str, CodeInterpreter]] = None,
    ) -> None:
        super().__init__(
            planner=(AzureOpenAILMM(temperature=0.0) if planner is None else planner),
            tool_recommender=(
                AzureSim(T.get_tools_df(), sim_key="desc")
                if tool_recommender is None
                else tool_recommender
            ),
            verbosity=verbosity,
            report_progress_callback=report_progress_callback,
            code_interpreter=code_interpreter,
        )
