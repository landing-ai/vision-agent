import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

from rich.console import Console
from rich.syntax import Syntax
from tabulate import tabulate

import vision_agent.tools as T
from vision_agent.agent import Agent
from vision_agent.agent.vision_agent_prompts import (
    CODE,
    FEEDBACK,
    FIX_BUG,
    FULL_TASK,
    PLAN,
    REFLECT,
    SIMPLE_TEST,
    USER_REQ,
)
from vision_agent.llm import LLM, OpenAILLM
from vision_agent.lmm import LMM, OpenAILMM
from vision_agent.utils import Execute
from vision_agent.utils.sim import Sim

logging.basicConfig(stream=sys.stdout)
_LOGGER = logging.getLogger(__name__)
_MAX_TABULATE_COL_WIDTH = 80
_EXECUTE = Execute(600)
_CONSOLE = Console()
_DEFAULT_IMPORT = "\n".join(T.__new_tools__)


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
    model: Union[LLM, LMM],
    media: Optional[List[Union[str, Path]]] = None,
) -> List[Dict[str, str]]:
    chat = copy.deepcopy(chat)
    if chat[-1]["role"] != "user":
        raise ValueError("Last chat message must be from the user.")

    user_request = chat[-1]["content"]
    context = USER_REQ.format(user_request=user_request)
    prompt = PLAN.format(context=context, tool_desc=tool_desc, feedback=working_memory)
    chat[-1]["content"] = prompt
    if isinstance(model, OpenAILMM):
        return extract_json(model.chat(chat, images=media))["plan"]  # type: ignore
    else:
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
    log_progress: Callable[[Dict[str, Any]], None],
    verbosity: int = 0,
    max_retries: int = 3,
    input_media: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    log_progress(
        {
            "type": "code",
            "status": "started",
        }
    )
    code = extract_code(
        coder(CODE.format(docstring=tool_info, question=task, feedback=working_memory))
    )
    test = extract_code(
        tester(
            SIMPLE_TEST.format(
                docstring=tool_utils,
                question=task,
                code=code,
                feedback=working_memory,
                media=input_media,
            )
        )
    )

    log_progress(
        {
            "type": "code",
            "status": "running",
            "payload": {
                "code": code,
                "test": test,
            },
        }
    )
    success, result = _EXECUTE.run_isolation(f"{_DEFAULT_IMPORT}\n{code}\n{test}")
    log_progress(
        {
            "type": "code",
            "status": "completed" if success else "failed",
            "payload": {
                "code": code,
                "test": test,
                "result": result,
            },
        }
    )
    if verbosity == 2:
        _LOGGER.info("Initial code and tests:")
        _CONSOLE.print(
            Syntax(f"{code}\n{test}", "python", theme="gruvbox-dark", line_numbers=True)
        )
        _LOGGER.info(f"Initial result: {result}")

    count = 0
    new_working_memory = []
    while not success and count < max_retries:
        log_progress(
            {
                "type": "code",
                "status": "started",
            }
        )
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
        log_progress(
            {
                "type": "code",
                "status": "running",
                "payload": {
                    "code": code,
                    "test": test,
                },
            }
        )
        new_working_memory.append(
            {"code": f"{code}\n{test}", "feedback": fixed_code_and_test["reflections"]}
        )

        success, result = _EXECUTE.run_isolation(f"{_DEFAULT_IMPORT}\n{code}\n{test}")
        log_progress(
            {
                "type": "code",
                "status": "completed" if success else "failed",
                "payload": {
                    "code": code,
                    "test": test,
                    "result": result,
                },
            }
        )
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

    if verbosity >= 1:
        _LOGGER.info("Final code and tests:")
        _CONSOLE.print(
            Syntax(f"{code}\n{test}", "python", theme="gruvbox-dark", line_numbers=True)
        )
        _LOGGER.info(f"Final Result: {result}")

    return {
        "code": code,
        "test": test,
        "success": success,
        "test_result": result,
        "working_memory": new_working_memory,
    }


def retrieve_tools(
    plan: List[Dict[str, str]],
    tool_recommender: Sim,
    log_progress: Callable[[Dict[str, Any]], None],
    verbosity: int = 0,
) -> str:
    log_progress(
        {
            "type": "tools",
            "status": "started",
        }
    )
    tool_info = []
    tool_desc = []
    for task in plan:
        tools = tool_recommender.top_k(task["instructions"], k=2, thresh=0.3)
        tool_info.extend([e["doc"] for e in tools])
        tool_desc.extend([e["desc"] for e in tools])
    log_progress(
        {
            "type": "tools",
            "status": "completed",
            "payload": tools,
        }
    )
    if verbosity == 2:
        _LOGGER.info(f"Tools: {tool_desc}")
    tool_info_set = set(tool_info)
    return "\n\n".join(tool_info_set)


class VisionAgent(Agent):
    """Vision Agent is an agentic framework that can output code based on a user
    request. It can plan tasks, retrieve relevant tools, write code, write tests and
    reflect on failed test cases to debug code. It is inspired by AgentCoder
    https://arxiv.org/abs/2312.13010 and Data Interpeter
    https://arxiv.org/abs/2402.18679

    Example
    -------
        >>> from vision_agent import VisionAgent
        >>> agent = VisionAgent()
        >>> code = agent("What percentage of the area of the jar is filled with coffee beans?", media="jar.jpg")
    """

    def __init__(
        self,
        planner: Optional[LLM] = None,
        coder: Optional[LLM] = None,
        tester: Optional[LLM] = None,
        debugger: Optional[LLM] = None,
        tool_recommender: Optional[Sim] = None,
        verbosity: int = 0,
        report_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Initialize the Vision Agent.

        Parameters:
            planner (Optional[LLM]): The planner model to use. Defaults to OpenAILLM.
            coder (Optional[LLM]): The coder model to use. Defaults to OpenAILLM.
            tester (Optional[LLM]): The tester model to use. Defaults to OpenAILLM.
            debugger (Optional[LLM]): The debugger model to
            tool_recommender (Optional[Sim]): The tool recommender model to use.
            verbosity (int): The verbosity level of the agent. Defaults to 0. 2 is the
                highest verbosity level which will output all intermediate debugging
                code.
            report_progress_callback: a callback to report the progress of the agent.
                This is useful for streaming logs in a web application where multiple
                VisionAgent instances are running in parallel. This callback ensures
                that the progress are not mixed up.
        """

        self.planner = (
            OpenAILLM(temperature=0.0, json_mode=True) if planner is None else planner
        )
        self.coder = OpenAILLM(temperature=0.0) if coder is None else coder
        self.tester = OpenAILLM(temperature=0.0) if tester is None else tester
        self.debugger = (
            OpenAILLM(temperature=0.0, json_mode=True) if debugger is None else debugger
        )

        self.tool_recommender = (
            Sim(T.TOOLS_DF, sim_key="desc")
            if tool_recommender is None
            else tool_recommender
        )
        self.verbosity = verbosity
        self.max_retries = 2
        self.report_progress_callback = report_progress_callback

    def __call__(
        self,
        input: Union[List[Dict[str, str]], str],
        media: Optional[Union[str, Path]] = None,
    ) -> str:
        """Chat with Vision Agent and return intermediate information regarding the task.

        Parameters:
            chat (List[Dict[str, str]]): A conversation in the format of
                [{"role": "user", "content": "describe your task here..."}].
            media (Optional[Union[str, Path]]): The media file to be used in the task.
            self_reflection (bool): Whether to reflect on the task and debug the code.

        Returns:
            str: The code output by the Vision Agent.
        """

        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        results = self.chat_with_workflow(input, media)
        results.pop("working_memory")
        return results  # type: ignore

    def chat_with_workflow(
        self,
        chat: List[Dict[str, str]],
        media: Optional[Union[str, Path]] = None,
        self_reflection: bool = False,
    ) -> Dict[str, Any]:
        """Chat with Vision Agent and return intermediate information regarding the task.

        Parameters:
            chat (List[Dict[str, str]]): A conversation in the format of
                [{"role": "user", "content": "describe your task here..."}].
            media (Optional[Union[str, Path]]): The media file to be used in the task.
            self_reflection (bool): Whether to reflect on the task and debug the code.

        Returns:
            Dict[str, Any]: A dictionary containing the code, test, test result, plan,
                and working memory of the agent.
        """

        if len(chat) == 0:
            raise ValueError("Chat cannot be empty.")

        if media is not None:
            for chat_i in chat:
                if chat_i["role"] == "user":
                    chat_i["content"] += f" Image name {media}"

        # re-grab custom tools
        global _DEFAULT_IMPORT
        _DEFAULT_IMPORT = "\n".join(T.__new_tools__)

        code = ""
        test = ""
        working_memory: List[Dict[str, str]] = []
        results = {"code": "", "test": "", "plan": []}
        plan = []
        success = False
        retries = 0

        while not success and retries < self.max_retries:
            self.log_progress(
                {
                    "type": "plans",
                    "status": "started",
                }
            )
            plan_i = write_plan(
                chat,
                T.TOOL_DESCRIPTIONS,
                format_memory(working_memory),
                self.planner,
                media=[media] if media else None,
            )
            plan_i_str = "\n-".join([e["instructions"] for e in plan_i])

            self.log_progress(
                {
                    "type": "plans",
                    "status": "completed",
                    "payload": plan_i,
                }
            )
            if self.verbosity >= 1:

                _LOGGER.info(
                    f"""
{tabulate(tabular_data=plan_i, headers="keys", tablefmt="mixed_grid", maxcolwidths=_MAX_TABULATE_COL_WIDTH)}"""
                )

            tool_info = retrieve_tools(
                plan_i,
                self.tool_recommender,
                self.log_progress,
                self.verbosity,
            )
            results = write_and_test_code(
                FULL_TASK.format(user_request=chat[0]["content"], subtasks=plan_i_str),
                tool_info,
                T.UTILITIES_DOCSTRING,
                format_memory(working_memory),
                self.coder,
                self.tester,
                self.debugger,
                self.log_progress,
                verbosity=self.verbosity,
                input_media=media,
            )
            success = cast(bool, results["success"])
            code = cast(str, results["code"])
            test = cast(str, results["test"])
            working_memory.extend(results["working_memory"])  # type: ignore
            plan.append({"code": code, "test": test, "plan": plan_i})

            if self_reflection:
                self.log_progress(
                    {
                        "type": "self_reflection",
                        "status": "started",
                    }
                )
                reflection = reflect(
                    chat,
                    FULL_TASK.format(
                        user_request=chat[0]["content"], subtasks=plan_i_str
                    ),
                    code,
                    self.planner,
                )
                if self.verbosity > 0:
                    _LOGGER.info(f"Reflection: {reflection}")
                feedback = cast(str, reflection["feedback"])
                success = cast(bool, reflection["success"])
                self.log_progress(
                    {
                        "type": "self_reflection",
                        "status": "completed" if success else "failed",
                        "payload": reflection,
                    }
                )
                working_memory.append({"code": f"{code}\n{test}", "feedback": feedback})

            retries += 1

        self.log_progress(
            {
                "type": "final_code",
                "status": "completed" if success else "failed",
                "payload": {
                    "code": code,
                    "test": test,
                    "result": results["test_result"],
                },
            }
        )

        return {
            "code": code,
            "test": test,
            "test_result": results["test_result"],
            "plan": plan,
            "working_memory": working_memory,
        }

    def log_progress(self, data: Dict[str, Any]) -> None:
        if self.report_progress_callback is not None:
            self.report_progress_callback(data)
        pass
