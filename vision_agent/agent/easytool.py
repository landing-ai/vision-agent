import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from vision_agent.llm import LLM, OpenAILLM
from vision_agent.lmm import LMM
from vision_agent.tools import TOOLS

from .agent import Agent
from .easytool_prompts import (
    ANSWER_GENERATE,
    ANSWER_SUMMARIZE,
    CHOOSE_PARAMETER,
    CHOOSE_TOOL,
    TASK_DECOMPOSE,
    TASK_TOPOLOGY,
)

logging.basicConfig(stream=sys.stdout)
_LOGGER = logging.getLogger(__name__)


def parse_json(s: str) -> Any:
    s = (
        s.replace(": true", ": True")
        .replace(": false", ": False")
        .replace(":true", ": True")
        .replace(":false", ": False")
        .replace("```", "")
        .strip()
    )
    return json.loads(s)


def change_name(name: str) -> str:
    change_list = ["from", "class", "return", "false", "true", "id", "and", "", "ID"]
    if name in change_list:
        name = "is_" + name.lower()
    return name


def format_tools(tools: Dict[int, Any]) -> str:
    # Format this way so it's clear what the ID's are
    tool_str = ""
    for key in tools:
        tool_str += f"ID: {key} - {tools[key]}\n"
    return tool_str


def topological_sort(tasks: List[Dict]) -> List[Dict]:
    in_degree = {task["id"]: 0 for task in tasks}
    for task in tasks:
        for dep in task["dep"]:
            if dep in in_degree:
                in_degree[task["id"]] += 1

    queue = [task for task in tasks if in_degree[task["id"]] == 0]
    sorted_order = []

    while queue:
        current = queue.pop(0)
        sorted_order.append(current)

        for task in tasks:
            if current["id"] in task["dep"]:
                in_degree[task["id"]] -= 1
                if in_degree[task["id"]] == 0:
                    queue.append(task)

    if len(sorted_order) != len(tasks):
        completed_ids = set([task["id"] for task in sorted_order])
        remaining_tasks = [task for task in tasks if task["id"] not in completed_ids]
        sorted_order.extend(remaining_tasks)
    return sorted_order


def task_decompose(
    model: Union[LLM, LMM, Agent], question: str, tools: Dict[int, Any]
) -> Optional[Dict]:
    prompt = TASK_DECOMPOSE.format(question=question, tools=format_tools(tools))
    tries = 0
    str_result = ""
    while True:
        try:
            str_result = model(prompt)
            result = parse_json(str_result)
            return result["Tasks"]  # type: ignore
        except Exception:
            if tries > 10:
                _LOGGER.error(f"Failed task_decompose on: {str_result}")
                return None
            tries += 1
            continue


def task_topology(
    model: Union[LLM, LMM, Agent], question: str, task_list: List[Dict]
) -> List[Dict[str, Any]]:
    prompt = TASK_TOPOLOGY.format(question=question, task_list=task_list)
    tries = 0
    str_result = ""
    while True:
        try:
            str_result = model(prompt)
            result = parse_json(str_result)
            for elt in result["Tasks"]:
                if isinstance(elt["dep"], str):
                    elt["dep"] = [int(dep) for dep in elt["dep"].split(",")]
                elif isinstance(elt["dep"], int):
                    elt["dep"] = [elt["dep"]]
                elif isinstance(elt["dep"], list):
                    elt["dep"] = [int(dep) for dep in elt["dep"]]
            return result["Tasks"]  # type: ignore
        except Exception:
            if tries > 10:
                _LOGGER.error(f"Failed task_topology on: {str_result}")
                return task_list
            tries += 1
            continue


def choose_tool(
    model: Union[LLM, LMM, Agent], question: str, tools: Dict[int, Any]
) -> Optional[int]:
    prompt = CHOOSE_TOOL.format(question=question, tools=format_tools(tools))
    tries = 0
    str_result = ""
    while True:
        try:
            str_result = model(prompt)
            result = parse_json(str_result)
            return result["ID"]  # type: ignore
        except Exception:
            if tries > 10:
                _LOGGER.error(f"Failed choose_tool on: {str_result}")
                return None
            tries += 1
            continue


def choose_parameter(
    model: Union[LLM, LMM, Agent], question: str, tool_usage: Dict, previous_log: str
) -> Optional[Any]:
    # TODO: should format tool_usage
    prompt = CHOOSE_PARAMETER.format(
        question=question, tool_usage=tool_usage, previous_log=previous_log
    )
    tries = 0
    str_result = ""
    while True:
        try:
            str_result = model(prompt)
            result = parse_json(str_result)
            return result["Parameters"]
        except Exception:
            if tries > 10:
                _LOGGER.error(f"Failed choose_parameter on: {str_result}")
                return None
            tries += 1
            continue


def answer_generate(
    model: Union[LLM, LMM, Agent], question: str, call_results: str, previous_log: str
) -> str:
    prompt = ANSWER_GENERATE.format(
        question=question, call_results=call_results, previous_log=previous_log
    )
    return model(prompt)


def answer_summarize(
    model: Union[LLM, LMM, Agent], question: str, answers: List[Dict]
) -> str:
    prompt = ANSWER_SUMMARIZE.format(question=question, answers=answers)
    return model(prompt)


def function_call(tool: Callable, parameters: Dict[str, Any]) -> Any:
    try:
        return tool()(**parameters)
    except Exception as e:
        _LOGGER.error(f"Failed function_call on: {e}")
        return None


def retrieval(
    model: Union[LLM, LMM, Agent],
    question: str,
    tools: Dict[int, Any],
    previous_log: str,
) -> Tuple[List[Dict], str]:
    tool_id = choose_tool(
        model, question, {k: v["description"] for k, v in tools.items()}
    )
    if tool_id is None:
        return [{}], ""
    _LOGGER.info(f"\t(Tool ID, name): ({tool_id}, {tools[tool_id]['name']})")

    tool_instructions = tools[tool_id]
    tool_usage = tool_instructions["usage"]
    tool_name = tool_instructions["name"]

    parameters = choose_parameter(model, question, tool_usage, previous_log)
    _LOGGER.info(f"\tParameters: {parameters} for {tool_name}")
    if parameters is None:
        return [{}], ""
    tool_results = [
        {"task": question, "tool_name": tool_name, "parameters": parameters}
    ]

    def parse_tool_results(result: Dict[str, Union[Dict, List]]) -> Any:
        call_results: List[Any] = []
        if isinstance(result["parameters"], Dict):
            call_result = function_call(tools[tool_id]["class"], result["parameters"])
            if call_result is None:
                return call_results
            call_results.append(call_result)
        elif isinstance(result["parameters"], List):
            for parameters in result["parameters"]:
                call_result = function_call(tools[tool_id]["class"], parameters)
                if call_result is None:
                    continue
                call_results.append(call_result)
        return call_results

    call_results = []
    for i, result in enumerate(tool_results):
        call_results.extend(parse_tool_results(result))
        tool_results[i]["call_results"] = call_results

    call_results_str = "\n\n".join([str(e) for e in call_results if e is not None])
    _LOGGER.info(f"\tCall Results: {call_results_str}")
    return tool_results, call_results_str


class EasyTool(Agent):
    r"""This is an implementation of the EasyTool paper https://arxiv.org/abs/2401.06201
    based on the original implementation https://github.com/microsoft/JARVIS/tree/main/easytool
    from the funcQA code.

    Example
    -------
        >>> from vision_agent.agent import EasyTool
        >>> agent = EasyTool()
        >>> resp = agent("If a car is traveling at 64 km/h, how many kilometers does it travel in 29 minutes?")
        >>> print(resp)
        "It will travel approximately 31.03 kilometers in 29 minutes."
        >>> resp = agent("How many cards are in this image?", image="cards.jpg")
        >>> print(resp)
        "There are 2 cards in this image."
    """

    def __init__(
        self,
        task_model: Optional[Union[LLM, LMM]] = None,
        answer_model: Optional[Union[LLM, LMM]] = None,
        verbose: bool = False,
    ):
        self.task_model = (
            OpenAILLM(json_mode=True) if task_model is None else task_model
        )
        self.answer_model = OpenAILLM() if answer_model is None else answer_model

        self.retrieval_num = 3
        self.tools = TOOLS
        if verbose:
            _LOGGER.setLevel(logging.INFO)

    def __call__(
        self,
        input: Union[List[Dict[str, str]], str],
        image: Optional[Union[str, Path]] = None,
    ) -> str:
        """Invoke the vision agent.

        Parameters:
            input: a prompt that describe the task or a conversation in the format of [{"role": "user", "content": "describe your task here..."}].
            image: the input image referenced in the prompt parameter.

        Returns:
            A text response.
        """
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        return self.chat(input, image=image)

    def chat_with_workflow(
        self, chat: List[Dict[str, str]], image: Optional[Union[str, Path]] = None
    ) -> Tuple[str, List[Dict]]:
        question = chat[0]["content"]
        if image:
            question += f" Image name: {image}"
        tasks = task_decompose(
            self.task_model,
            question,
            {k: v["description"] for k, v in self.tools.items()},
        )
        _LOGGER.info(f"Tasks: {tasks}")
        if tasks is not None:
            task_list = [{"task": task, "id": i + 1} for i, task in enumerate(tasks)]
            task_list = task_topology(self.task_model, question, task_list)
            try:
                task_list = topological_sort(task_list)
            except Exception:
                _LOGGER.error(f"Failed topological_sort on: {task_list}")
        else:
            task_list = []

        _LOGGER.info(f"Task Dependency: {task_list}")
        task_depend = {"Original Quesiton": question}
        previous_log = ""
        answers = []
        for task in task_list:
            task_depend[task["id"]] = {"task": task["task"], "answer": ""}  # type: ignore
        all_tool_results = []
        for task in task_list:
            task_str = task["task"]
            previous_log = str(task_depend)
            _LOGGER.info(f"\tSubtask: {task_str}")
            tool_results, call_results = retrieval(
                self.task_model,
                task_str,
                self.tools,
                previous_log,
            )
            answer = answer_generate(
                self.answer_model, task_str, call_results, previous_log
            )

            for tool_result in tool_results:
                tool_result["answer"] = answer
            all_tool_results.extend(tool_results)

            _LOGGER.info(f"\tAnswer: {answer}")
            answers.append({"task": task_str, "answer": answer})
            task_depend[task["id"]]["answer"] = answer  # type: ignore
        return answer_summarize(self.answer_model, question, answers), all_tool_results

    def chat(
        self, chat: List[Dict[str, str]], image: Optional[Union[str, Path]] = None
    ) -> str:
        answer, _ = self.chat_with_workflow(chat, image=image)
        return answer
