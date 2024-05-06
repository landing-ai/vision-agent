import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from rich.console import Console
from rich.syntax import Syntax
from tabulate import tabulate

from vision_agent.agent import Agent
from vision_agent.llm import LLM, OpenAILLM
from vision_agent.tools.tools_v2 import TOOL_DESCRIPTIONS, TOOLS_DF
from vision_agent.utils import Execute, Sim

from .automated_vision_agent_prompt import (
    CODE,
    CODE_SYS_MSG,
    DEBUG,
    DEBUG_EXAMPLE,
    DEBUG_SYS_MSG,
    PLAN,
    PREV_CODE_CONTEXT,
    PREV_CODE_CONTEXT_WITH_REFLECTION,
    TEST,
    USER_REQ_CONTEXT,
    USER_REQ_SUBTASK_CONTEXT,
)

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)
_MAX_TABULATE_COL_WIDTH = 80
_CONSOLE = Console()


def extract_code(code: str) -> str:
    if "```python" in code:
        code = code[code.find("```python") + len("```python") :]
        code = code[: code.find("```")]
    return code


def write_plan(
    user_requirements: str, tool_desc: str, model: LLM
) -> List[Dict[str, Any]]:
    context = USER_REQ_CONTEXT.format(user_requirement=user_requirements)
    prompt = PLAN.format(context=context, plan="", tool_desc=tool_desc)
    plan = json.loads(model(prompt).replace("```", "").strip())
    return plan["plan"]


def write_code(user_req: str, subtask: str, tool_info: str, code: str, model: LLM) -> str:
    prompt = CODE.format(
        context=USER_REQ_SUBTASK_CONTEXT.format(user_requirement=user_req, subtask=subtask),
        tool_info=tool_info,
        code=code,
    )
    messages = [
        {"role": "system", "content": CODE_SYS_MSG},
        {"role": "user", "content": prompt},
    ]
    code = model.chat(messages)
    return extract_code(code)


def write_test(user_req: str, subtask: str, tool_info: str, code: str, model: LLM) -> str:
    prompt = TEST.format(
        context=USER_REQ_SUBTASK_CONTEXT.format(user_requirement=user_req, subtask=subtask),
        tool_info=tool_info,
        code=code,
    )
    messages = [
        {"role": "system", "content": CODE_SYS_MSG},
        {"role": "user", "content": prompt},
    ]
    code = model.chat(messages)
    return extract_code(code)


def debug_code(sub_task: str, working_memory: List[str], model: LLM) -> Tuple[str, str]:
    # Make debug model output JSON
    if hasattr(model, "kwargs"):
        model.kwargs["response_format"] = {"type": "json_object"}
    prompt = DEBUG.format(
        debug_example=DEBUG_EXAMPLE,
        context=USER_REQ_CONTEXT.format(user_requirement=sub_task),
        previous_impl="\n".join(working_memory),
    )
    messages = [
        {"role": "system", "content": DEBUG_SYS_MSG},
        {"role": "user", "content": prompt},
    ]
    code_and_ref = json.loads(model.chat(messages).replace("```", "").strip())
    if hasattr(model, "kwargs"):
        del model.kwargs["response_format"]
    return extract_code(code_and_ref["improved_impl"]), code_and_ref["reflection"]


def write_and_exec_code(
    user_req: str,
    subtask: str,
    orig_code: str,
    code_writer_call: Callable,
    model: LLM,
    tool_info: str,
    exec: Execute,
    max_retry: int = 3,
    verbose: bool = False,
) -> Tuple[bool, str, str, Dict[str, List[str]]]:
    success = False
    counter = 0
    reflection = ""

    # TODO: add working memory to code_writer_call and debug_code
    code = code_writer_call(user_req, subtask, tool_info, orig_code, model)
    success, result = exec.run_isolation(code)
    working_memory: Dict[str, List[str]] = {}
    while not success and counter < max_retry:
        if subtask not in working_memory:
            working_memory[subtask] = []

        if reflection:
            working_memory[subtask].append(
                PREV_CODE_CONTEXT_WITH_REFLECTION.format(
                    code=code, result=result, reflection=reflection
                )
            )
        else:
            working_memory[subtask].append(
                PREV_CODE_CONTEXT.format(code=code, result=result)
            )

        code, reflection = debug_code(subtask, working_memory[subtask], model)
        success, result = exec.run_isolation(code)
        counter += 1
        if verbose:
            _CONSOLE.print(
                Syntax(code, "python", theme="gruvbox-dark", line_numbers=True)
            )
        _LOGGER.info(f"\tDebugging reflection, result: {reflection}, {result}")

        if success:
            working_memory[subtask].append(
                PREV_CODE_CONTEXT_WITH_REFLECTION.format(
                    code=code, result=result, reflection=reflection
                )
            )

    return success, code, result, working_memory


def run_plan(
    user_req: str,
    plan: List[Dict[str, Any]],
    coder: LLM,
    exec: Execute,
    code: str,
    tool_recommender: Sim,
    verbose: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    active_plan = [e for e in plan if "success" not in e or not e["success"]]
    working_memory: Dict[str, List[str]] = {}
    for task in active_plan:
        _LOGGER.info(
            f"""
{tabulate(tabular_data=[task], headers="keys", tablefmt="mixed_grid", maxcolwidths=_MAX_TABULATE_COL_WIDTH)}"""
        )
        tool_info = "\n".join([e["doc"] for e in tool_recommender.top_k(task["instruction"])])
        success, code, result, task_memory = write_and_exec_code(
            user_req,
            task["instruction"],
            code,
            write_code if task["type"] == "code" else write_test,
            coder,
            tool_info,
            exec,
            verbose,
        )
        working_memory.update(task_memory)

        if verbose:
            _CONSOLE.print(
                Syntax(code, "python", theme="gruvbox-dark", line_numbers=True)
            )
        _LOGGER.info(f"\tCode success, result: {success}, {str(result)}")

        task["success"] = success
        task["result"] = result
        task["code"] = code

        if not success:
            break

    return code, plan


class AutomatedVisionAgent(Agent):
    def __init__(
        self,
        timeout: int = 600,
        tool_recommender: Optional[Sim] = None,
        verbose: bool = False,
    ) -> None:
        self.planner = OpenAILLM(temperature=0.1, json_mode=True)
        self.coder = OpenAILLM(temperature=0.1)
        self.exec = Execute(timeout=timeout)
        if tool_recommender is None:
            self.tool_recommender = Sim(TOOLS_DF, sim_key="desc")
        else:
            self.tool_recommender = tool_recommender
        self.long_term_memory = []
        self.verbose = verbose
        if self.verbose:
            _LOGGER.setLevel(logging.INFO)

    def __call__(
        self,
        input: Union[List[Dict[str, str]], str],
        image: Optional[Union[str, Path]] = None,
    ) -> str:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        return self.chat(input, image)

    def chat(
        self,
        chat: List[Dict[str, str]],
        image: Optional[Union[str, Path]] = None,
    ) -> str:
        if len(chat) == 0:
            raise ValueError("Input cannot be empty.")

        user_req = chat[0]["content"]
        if image is not None:
            user_req += f" Image name {image}"

        plan = write_plan(user_req, TOOL_DESCRIPTIONS, self.planner)
        _LOGGER.info(
            f"""Plan:
{tabulate(tabular_data=plan, headers="keys", tablefmt="mixed_grid", maxcolwidths=_MAX_TABULATE_COL_WIDTH)}"""
        )
        working_memory: Dict[str, List[str]] = {}

        working_code = ""
        working_test = ""
        success = False

        while not success:
            working_code, plan = run_plan(
                user_req,
                plan,
                self.coder,
                self.exec,
                working_code,
                self.tool_recommender,
                self.verbose,
            )
            success = all(task["success"] for task in plan)

            if not success:
                pass

        return working_code

    def log_progress(self, description: str) -> None:
        pass
