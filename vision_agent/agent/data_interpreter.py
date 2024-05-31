import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import pandas as pd
from langsmith import traceable
from rich.console import Console
from rich.syntax import Syntax
from tabulate import tabulate

from vision_agent.agent import Agent
from vision_agent.agent.data_interpreter_prompts import (
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
    USER_REQ_SUBTASK_WM_CONTEXT,
)
from vision_agent.llm import LLM, OpenAILLM
from vision_agent.tools import TOOL_DESCRIPTIONS, TOOLS_DF
from vision_agent.utils import CodeInterpreter, CodeInterpreterFactory, Execution, Sim

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)
_MAX_TABULATE_COL_WIDTH = 80
_EXECUTE = CodeInterpreterFactory.get_default_instance()
_CONSOLE = Console()


def build_working_memory(working_memory: Mapping[str, List[str]]) -> Sim:
    data: Mapping[str, List[str]] = {"desc": [], "doc": []}
    for key, value in working_memory.items():
        data["desc"].append(key)
        data["doc"].append("\n".join(value))
    df = pd.DataFrame(data)  # type: ignore
    return Sim(df, sim_key="desc")


def extract_code(code: str) -> str:
    if "```python" in code:
        code = code[code.find("```python") + len("```python") :]
        code = code[: code.find("```")]
    if code.startswith("python\n"):
        code = code[len("python\n") :]
    return code


def extract_json(json_str: str) -> Dict[str, Any]:
    try:
        json_dict = json.loads(json_str)
    except json.JSONDecodeError:
        if "```json" in json_str:
            json_str = json_str[json_str.find("```json") + len("```json") :]
            json_str = json_str[: json_str.find("```")]
        elif "```" in json_str:
            json_str = json_str[json_str.find("```") + len("```") :]
            # get the last ``` not one from an intermediate string
            json_str = json_str[: json_str.find("}```")]
        json_dict = json.loads(json_str)
    return json_dict  # type: ignore


@traceable(name="planning")
def write_plan(
    chat: List[Dict[str, str]],
    plan: Optional[List[Dict[str, Any]]],
    tool_desc: str,
    model: LLM,
) -> Tuple[str, List[Dict[str, Any]]]:
    # Get last user request
    if chat[-1]["role"] != "user":
        raise ValueError("Last chat message must be from the user.")
    user_requirements = chat[-1]["content"]

    context = USER_REQ_CONTEXT.format(user_requirement=user_requirements)
    prompt = PLAN.format(context=context, plan=str(plan), tool_desc=tool_desc)
    chat[-1]["content"] = prompt
    new_plan = extract_json(model.chat(chat))
    return new_plan["user_req"], new_plan["plan"]


def write_code(
    user_req: str,
    subtask: str,
    working_memory: str,
    tool_info: str,
    code: str,
    model: LLM,
) -> str:
    prompt = CODE.format(
        context=USER_REQ_SUBTASK_WM_CONTEXT.format(
            user_requirement=user_req, working_memory=working_memory, subtask=subtask
        ),
        tool_info=tool_info,
        code=code,
    )
    messages = [
        {"role": "system", "content": CODE_SYS_MSG},
        {"role": "user", "content": prompt},
    ]
    code = model.chat(messages)
    return extract_code(code)


def write_test(
    user_req: str, subtask: str, tool_info: str, _: str, code: str, model: LLM
) -> str:
    prompt = TEST.format(
        context=USER_REQ_SUBTASK_CONTEXT.format(
            user_requirement=user_req, subtask=subtask
        ),
        tool_info=tool_info,
        code=code,
    )
    messages = [
        {"role": "system", "content": CODE_SYS_MSG},
        {"role": "user", "content": prompt},
    ]
    code = model.chat(messages)
    return extract_code(code)


def debug_code(
    user_req: str,
    subtask: str,
    retrieved_ltm: str,
    working_memory: str,
    model: LLM,
) -> Tuple[str, str]:
    # Make debug model output JSON
    if hasattr(model, "kwargs"):
        model.kwargs["response_format"] = {"type": "json_object"}
    prompt = DEBUG.format(
        debug_example=DEBUG_EXAMPLE,
        context=USER_REQ_SUBTASK_WM_CONTEXT.format(
            user_requirement=user_req,
            subtask=subtask,
            working_memory=retrieved_ltm,
        ),
        previous_impl=working_memory,
    )
    messages = [
        {"role": "system", "content": DEBUG_SYS_MSG},
        {"role": "user", "content": prompt},
    ]
    code_and_ref = extract_json(model.chat(messages))
    if hasattr(model, "kwargs"):
        del model.kwargs["response_format"]
    return extract_code(code_and_ref["improved_impl"]), code_and_ref["reflection"]


def write_and_exec_code(
    user_req: str,
    subtask: str,
    orig_code: str,
    code_writer_call: Callable[..., str],
    model: LLM,
    tool_info: str,
    exec: CodeInterpreter,
    retrieved_ltm: str,
    log_progress: Callable[[Dict[str, Any]], None],
    max_retry: int = 3,
    verbosity: int = 0,
) -> Tuple[bool, str, Execution, Dict[str, List[str]]]:
    success = False
    counter = 0
    reflection = ""

    code = code_writer_call(
        user_req, subtask, retrieved_ltm, tool_info, orig_code, model
    )
    result = exec.exec_isolation(code)
    success = result.success
    if verbosity == 2:
        _CONSOLE.print(Syntax(code, "python", theme="gruvbox-dark", line_numbers=True))
        log_progress(
            {
                "log": f"Code success: {success}",
            }
        )
        log_progress(
            {
                "log": "Code:",
                "code": code,
            }
        )
        log_progress(
            {
                "log": "Result:",
                "result": result.to_json(),
            }
        )
        _LOGGER.info(f"\tCode success: {success}, result: {result.text(False)}")
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
                PREV_CODE_CONTEXT.format(code=code, result=result.text())
            )

        code, reflection = debug_code(
            user_req, subtask, retrieved_ltm, "\n".join(working_memory[subtask]), model
        )
        result = exec.exec_isolation(code)
        counter += 1
        if verbosity == 2:
            _CONSOLE.print(
                Syntax(code, "python", theme="gruvbox-dark", line_numbers=True)
            )
            log_progress(
                {
                    "log": "Debugging reflection:",
                    "reflection": reflection,
                }
            )
            log_progress(
                {
                    "log": "Result:",
                    "result": result.to_json(),
                }
            )
            _LOGGER.info(
                f"\tDebugging reflection: {reflection}, result: {result.text(False)}"
            )

        if success:
            working_memory[subtask].append(
                PREV_CODE_CONTEXT_WITH_REFLECTION.format(
                    reflection=reflection, code=code, result=result.text()
                )
            )

    return result.success, code, result, working_memory


@traceable(name="plan execution")
def run_plan(
    user_req: str,
    plan: List[Dict[str, Any]],
    coder: LLM,
    exec: CodeInterpreter,
    code: str,
    tool_recommender: Sim,
    log_progress: Callable[[Dict[str, Any]], None],
    long_term_memory: Optional[Sim] = None,
    verbosity: int = 0,
) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, List[str]]]:
    active_plan = [e for e in plan if "success" not in e or not e["success"]]
    current_code = code
    current_test = ""
    retrieved_ltm = ""
    working_memory: Dict[str, List[str]] = {}

    for task in active_plan:
        log_progress(
            {"log": "Going to run the following task(s) in sequence:", "task": task}
        )
        _LOGGER.info(
            f"""
{tabulate(tabular_data=[task], headers="keys", tablefmt="mixed_grid", maxcolwidths=_MAX_TABULATE_COL_WIDTH)}"""
        )
        tools = tool_recommender.top_k(task["instruction"], thresh=0.3)
        tool_info = "\n".join([e["doc"] for e in tools])

        if verbosity == 2:
            log_progress({"log": f"Tools retrieved: {[e['desc'] for e in tools]}"})
            _LOGGER.info(f"Tools retrieved: {[e['desc'] for e in tools]}")

        if long_term_memory is not None:
            retrieved_ltm = "\n".join(
                [e["doc"] for e in long_term_memory.top_k(task["instruction"], 1)]
            )

        success, code, result, working_memory_i = write_and_exec_code(
            user_req,
            task["instruction"],
            current_code,
            write_code if task["type"] == "code" else write_test,
            coder,
            tool_info,
            exec,
            retrieved_ltm,
            log_progress,
            verbosity=verbosity,
        )
        if task["type"] == "code":
            current_code = code
        else:
            current_test = code

        working_memory.update(working_memory_i)

        if verbosity == 1:
            _CONSOLE.print(
                Syntax(code, "python", theme="gruvbox-dark", line_numbers=True)
            )

        log_progress(
            {
                "log": f"Code success: {success}",
            }
        )
        log_progress(
            {
                "log": "Result:",
                "result": result.to_json(),
            }
        )
        _LOGGER.info(f"\tCode success: {success} result: {result.text(False)}")

        task["success"] = success
        task["result"] = result
        task["code"] = code

        if not success:
            break

    return current_code, current_test, plan, working_memory


class DataInterpreter(Agent):
    """This version of Data Interpreter is an AI agentic framework geared towards
    outputting Python code to solve vision tasks. It is inspired by MetaGPT's Data
    Interpreter https://arxiv.org/abs/2402.18679. This version of Data Interpreter has
    several key features to help it generate code:

    - A planner to generate a plan of tasks to solve a user requirement. The planner
    can output code tasks or test tasks, where test tasks are used to verify the code.
    - Automatic debugging, if a task fails, the agent will attempt to debug the code
    using the failed output to fix it.
    - A tool recommender to recommend tools to use for a given task. LLM performance
    on tool retrieval starts to decrease as you add more tools, tool retrieval helps
    keep the number of tools to choose from low.
    - Memory retrieval, the agent can remember previous iterations on tasks to help it
    with new tasks.
    - Dynamic replanning, the agent can ask for feedback and replan remaining tasks
    based off of that feedback.
    """

    def __init__(
        self,
        timeout: int = 600,
        tool_recommender: Optional[Sim] = None,
        long_term_memory: Optional[Sim] = None,
        verbosity: int = 0,
        report_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.planner = OpenAILLM(temperature=0.0, json_mode=True)
        self.coder = OpenAILLM(temperature=0.0)
        self.exec = _EXECUTE
        self.report_progress_callback = report_progress_callback
        if tool_recommender is None:
            self.tool_recommender = Sim(TOOLS_DF, sim_key="desc")
        else:
            self.tool_recommender = tool_recommender
        self.verbosity = verbosity
        self._working_memory: Dict[str, List[str]] = {}
        if long_term_memory is not None:
            if "doc" not in long_term_memory.df.columns:
                raise ValueError("Long term memory must have a 'doc' column.")
        self.long_term_memory = long_term_memory
        self.max_retries = 3
        if self.verbosity:
            _LOGGER.setLevel(logging.INFO)

    def __call__(
        self,
        input: Union[List[Dict[str, str]], str],
        media: Optional[Union[str, Path]] = None,
        plan: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        results = self.chat_with_workflow(input, media, plan)
        return results["code"]  # type: ignore

    @traceable
    def chat_with_workflow(
        self,
        chat: List[Dict[str, str]],
        media: Optional[Union[str, Path]] = None,
        plan: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if len(chat) == 0:
            raise ValueError("Input cannot be empty.")

        if media is not None:
            # append file names to all user messages
            for chat_i in chat:
                if chat_i["role"] == "user":
                    chat_i["content"] += f" Image name {media}"

        working_code = ""
        if plan is not None:
            # grab the latest working code from a previous plan
            for task in plan:
                if "success" in task and "code" in task and task["success"]:
                    working_code = task["code"]

        user_req, plan = write_plan(chat, plan, TOOL_DESCRIPTIONS, self.planner)
        self.log_progress(
            {
                "log": "Plans:",
                "plan": plan,
            }
        )
        _LOGGER.info(
            f"""Plan:
{tabulate(tabular_data=plan, headers="keys", tablefmt="mixed_grid", maxcolwidths=_MAX_TABULATE_COL_WIDTH)}"""
        )

        working_test = ""
        working_memory: Dict[str, List[str]] = {}
        success = False
        retries = 0

        while not success and retries < self.max_retries:
            working_code, working_test, plan, working_memory_i = run_plan(
                user_req,
                plan,
                self.coder,
                self.exec,
                working_code,
                self.tool_recommender,
                self.log_progress,
                self.long_term_memory,
                self.verbosity,
            )
            success = all(
                task["success"] if "success" in task else False for task in plan
            )
            working_memory.update(working_memory_i)

            if not success:
                # return to user and request feedback
                break

            retries += 1

        self.log_progress(
            {
                "log": f"The Vision Agent V2 has concluded this chat.\nSuccess: {success}",
                "finished": True,
            }
        )

        return {
            "code": working_code,
            "test": working_test,
            "success": success,
            "working_memory": build_working_memory(working_memory),
            "plan": plan,
        }

    def log_progress(self, data: Dict[str, Any]) -> None:
        if self.report_progress_callback is not None:
            self.report_progress_callback(data)
        pass
