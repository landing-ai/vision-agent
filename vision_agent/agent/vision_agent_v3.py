import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from rich.console import Console
from rich.syntax import Syntax
from tabulate import tabulate

from vision_agent.agent import Agent
from vision_agent.agent.vision_agent_v3_prompts import (
    CODE,
    FEEDBACK,
    FIX_BUG,
    PLAN,
    REFLECT,
    SIMPLE_TEST,
    USER_REQ,
)
from vision_agent.llm import LLM, OpenAILLM
from vision_agent.tools.tools_v2 import TOOL_DESCRIPTIONS, TOOLS_DF, UTILITIES_DOCSTRING
from vision_agent.utils import Execute
from vision_agent.utils.sim import Sim

logging.basicConfig(stream=sys.stdout)
_LOGGER = logging.getLogger(__name__)
_MAX_TABULATE_COL_WIDTH = 80
_EXECUTE = Execute(600)
_CONSOLE = Console()


def format_memory(memory: List[Dict[str, str]]) -> str:
    return FEEDBACK.format(
        feedback="\n".join(
            [
                f"### Feedback {i}:\nCode: ```python\n{m['code']}\n```\nFeedback: {m['feedback']}\n"
                for i, m in enumerate(memory)
            ]
        )
    )


def extract_code(code: str) -> str:
    if "\n```python" in code:
        start = "\n```python"
    elif "```python" in code:
        start = "```python"
    else:
        return code

    code = code[code.find(start) + len(start) :]
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


def write_plan(
    chat: List[Dict[str, str]],
    tool_desc: str,
    working_memory: str,
    model: LLM,
) -> List[Dict[str, str]]:
    chat = copy.deepcopy(chat)
    if chat[-1]["role"] != "user":
        raise ValueError("Last chat message must be from the user.")

    user_request = chat[-1]["content"]
    context = USER_REQ.format(user_request=user_request)
    prompt = PLAN.format(context=context, tool_desc=tool_desc, feedback=working_memory)
    chat[-1]["content"] = prompt
    return extract_json(model.chat(chat))["plan"]  # type: ignore


def reflect(
    chat: List[Dict[str, str]],
    plan: str,
    code: str,
    model: LLM,
) -> Dict[str, Union[str, bool]]:
    chat = copy.deepcopy(chat)
    if chat[-1]["role"] != "user":
        raise ValueError("Last chat message must be from the user.")

    user_request = chat[-1]["content"]
    context = USER_REQ.format(user_request=user_request)
    prompt = REFLECT.format(context=context, plan=plan, code=code)
    chat[-1]["content"] = prompt
    return extract_json(model.chat(chat))


def write_and_test_code(
    task: str,
    tool_info: str,
    tool_utils: str,
    working_memory: str,
    coder: LLM,
    tester: LLM,
    debugger: LLM,
    verbosity: int = 0,
    max_retries: int = 3,
) -> Dict[str, Any]:
    code = extract_code(
        coder(CODE.format(docstring=tool_info, question=task, feedback=working_memory))
    )
    test = extract_code(
        tester(
            SIMPLE_TEST.format(
                docstring=tool_utils, question=task, code=code, feedback=working_memory
            )
        )
    )

    success, result = _EXECUTE.run_isolation(f"{code}\n{test}")
    if verbosity == 2:
        _LOGGER.info("First code and tests:")
        _CONSOLE.print(
            Syntax(f"{code}\n{test}", "python", theme="gruvbox-dark", line_numbers=True)
        )
        _LOGGER.info(f"First result: {result}")

    count = 0
    new_working_memory = []
    while not success and count < max_retries:
        fixed_code_and_test = extract_json(
            debugger(
                FIX_BUG.format(
                    code=code, tests=test, result=result, feedback=working_memory
                )
            )
        )
        if fixed_code_and_test["code"].strip() != "":
            code = extract_code(fixed_code_and_test["code"])
        if fixed_code_and_test["test"].strip() != "":
            test = extract_code(fixed_code_and_test["test"])
        new_working_memory.append(
            {"code": f"{code}\n{test}", "feedback": fixed_code_and_test["reflections"]}
        )

        success, result = _EXECUTE.run_isolation(f"{code}\n{test}")
        if verbosity == 2:
            _LOGGER.info(
                f"Debug attempt {count + 1}, reflection: {fixed_code_and_test['reflections']}"
            )
            _CONSOLE.print(
                Syntax(
                    f"{code}\n{test}", "python", theme="gruvbox-dark", line_numbers=True
                )
            )
            _LOGGER.info(f"Debug result: {result}")
        count += 1

    if verbosity == 1:
        _CONSOLE.print(
            Syntax(f"{code}\n{test}", "python", theme="gruvbox-dark", line_numbers=True)
        )
        _LOGGER.info(f"Result: {result}")

    return {
        "code": code,
        "test": test,
        "success": success,
        "working_memory": new_working_memory,
    }


def retrieve_tools(
    plan: List[Dict[str, str]], tool_recommender: Sim, verbosity: int = 0
) -> str:
    tool_info = []
    tool_desc = []
    for task in plan:
        tools = tool_recommender.top_k(task["instructions"], k=2, thresh=0.3)
        tool_info.extend([e["doc"] for e in tools])
        tool_desc.extend([e["desc"] for e in tools])
    if verbosity == 2:
        _LOGGER.info(f"Tools: {tool_desc}")
    tool_info_set = set(tool_info)
    return "\n\n".join(tool_info_set)


class VisionAgentV3(Agent):
    def __init__(
        self,
        timeout: int = 600,
        planner: Optional[LLM] = None,
        coder: Optional[LLM] = None,
        tester: Optional[LLM] = None,
        debugger: Optional[LLM] = None,
        tool_recommender: Optional[Sim] = None,
        verbosity: int = 0,
    ) -> None:
        self.planner = (
            OpenAILLM(temperature=0.0, json_mode=True) if planner is None else planner
        )
        self.coder = OpenAILLM(temperature=0.0) if coder is None else coder
        self.tester = OpenAILLM(temperature=0.0) if tester is None else tester
        self.debugger = (
            OpenAILLM(temperature=0.0, json_mode=True) if debugger is None else debugger
        )

        self.tool_recommender = (
            Sim(TOOLS_DF, sim_key="desc")
            if tool_recommender is None
            else tool_recommender
        )
        self.verbosity = verbosity
        self.max_retries = 3

    def __call__(
        self,
        input: Union[List[Dict[str, str]], str],
        image: Optional[Union[str, Path]] = None,
    ) -> str:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        results = self.chat_with_workflow(input, image)
        return results["code"]  # type: ignore

    def chat_with_workflow(
        self,
        chat: List[Dict[str, str]],
        image: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        if len(chat) == 0:
            raise ValueError("Chat cannot be empty.")

        if image is not None:
            for chat_i in chat:
                if chat_i["role"] == "user":
                    chat_i["content"] += f" Image name {image}"

        code = ""
        test = ""
        working_memory: List[Dict[str, str]] = []
        results = {"code": "", "test": "", "plan": []}
        plan = []
        success = False
        retries = 0

        while not success and retries < self.max_retries:
            plan_i = write_plan(
                chat, TOOL_DESCRIPTIONS, format_memory(working_memory), self.planner
            )
            plan_i_str = "\n-".join([e["instructions"] for e in plan_i])
            if self.verbosity == 1 or self.verbosity == 2:
                _LOGGER.info(
                    f"""
{tabulate(tabular_data=plan_i, headers="keys", tablefmt="mixed_grid", maxcolwidths=_MAX_TABULATE_COL_WIDTH)}"""
                )

            tool_info = retrieve_tools(
                plan_i,
                self.tool_recommender,
                self.verbosity,
            )
            results = write_and_test_code(
                plan_i_str,
                tool_info,
                UTILITIES_DOCSTRING,
                format_memory(working_memory),
                self.coder,
                self.tester,
                self.debugger,
                verbosity=self.verbosity,
            )
            success = cast(bool, results["success"])
            code = cast(str, results["code"])
            test = cast(str, results["test"])
            working_memory.extend(results["working_memory"])  # type: ignore
            plan.append({"code": code, "test": test, "plan": plan_i})

            reflection = reflect(chat, plan_i_str, code, self.planner)
            if self.verbosity > 0:
                _LOGGER.info(f"Reflection: {reflection}")
            feedback = cast(str, reflection["feedback"])
            success = cast(bool, reflection["success"])
            working_memory.append({"code": f"{code}\n{test}", "feedback": feedback})

        return {
            "code": code,
            "test": test,
            "plan": plan,
            "working_memory": working_memory,
        }

    def log_progress(self, description: str) -> None:
        pass
