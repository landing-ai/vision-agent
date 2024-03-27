import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from tabulate import tabulate

from vision_agent.llm import LLM, OpenAILLM
from vision_agent.lmm import LMM, OpenAILMM
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
from .vision_agent_prompts import (
    ANSWER_GENERATE_DEPENDS,
    ANSWER_SUMMARIZE_DEPENDS,
    CHOOSE_PARAMETER_DEPENDS,
    CHOOSE_TOOL_DEPENDS,
    TASK_DECOMPOSE_DEPENDS,
    VISION_AGENT_REFLECTION,
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
    model: Union[LLM, LMM, Agent],
    question: str,
    tools: Dict[int, Any],
    reflections: str,
) -> Optional[Dict]:
    if reflections:
        prompt = TASK_DECOMPOSE_DEPENDS.format(
            question=question, tools=format_tools(tools), reflections=reflections
        )
    else:
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
    model: Union[LLM, LMM, Agent],
    question: str,
    tools: Dict[int, Any],
    reflections: str,
) -> Optional[int]:
    if reflections:
        prompt = CHOOSE_TOOL_DEPENDS.format(
            question=question, tools=format_tools(tools), reflections=reflections
        )
    else:
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
    model: Union[LLM, LMM, Agent],
    question: str,
    tool_usage: Dict,
    previous_log: str,
    reflections: str,
) -> Optional[Any]:
    # TODO: should format tool_usage
    if reflections:
        prompt = CHOOSE_PARAMETER_DEPENDS.format(
            question=question,
            tool_usage=tool_usage,
            previous_log=previous_log,
            reflections=reflections,
        )
    else:
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
    model: Union[LLM, LMM, Agent],
    question: str,
    call_results: str,
    previous_log: str,
    reflections: str,
) -> str:
    if reflections:
        prompt = ANSWER_GENERATE_DEPENDS.format(
            question=question,
            call_results=call_results,
            previous_log=previous_log,
            reflections=reflections,
        )
    else:
        prompt = ANSWER_GENERATE.format(
            question=question, call_results=call_results, previous_log=previous_log
        )
    return model(prompt)


def answer_summarize(
    model: Union[LLM, LMM, Agent], question: str, answers: List[Dict], reflections: str
) -> str:
    if reflections:
        prompt = ANSWER_SUMMARIZE_DEPENDS.format(
            question=question, answers=answers, reflections=reflections
        )
    else:
        prompt = ANSWER_SUMMARIZE.format(question=question, answers=answers)
    return model(prompt)


def function_call(tool: Callable, parameters: Dict[str, Any]) -> Any:
    try:
        return tool()(**parameters)
    except Exception as e:
        _LOGGER.error(f"Failed function_call on: {e}")
        # return error message so it can self-correct
        return str(e)


def retrieval(
    model: Union[LLM, LMM, Agent],
    question: str,
    tools: Dict[int, Any],
    previous_log: str,
    reflections: str,
) -> Tuple[List[Dict], str]:
    tool_id = choose_tool(
        model, question, {k: v["description"] for k, v in tools.items()}, reflections
    )
    if tool_id is None:
        return [{}], ""
    _LOGGER.info(f"\t(Tool ID, name): ({tool_id}, {tools[tool_id]['name']})")

    tool_instructions = tools[tool_id]
    tool_usage = tool_instructions["usage"]
    tool_name = tool_instructions["name"]

    parameters = choose_parameter(
        model, question, tool_usage, previous_log, reflections
    )
    _LOGGER.info(f"\tParameters: {parameters} for {tool_name}")
    if parameters is None:
        return [{}], ""
    tool_results = [
        {"task": question, "tool_name": tool_name, "parameters": parameters}
    ]

    _LOGGER.info(
        f"""Going to run the following {len(tool_results)} tool(s) in sequence:
{tabulate(tool_results, headers="keys", tablefmt="mixed_grid")}"""
    )

    def parse_tool_results(result: Dict[str, Union[Dict, List]]) -> Any:
        call_results: List[Any] = []
        if isinstance(result["parameters"], Dict):
            call_results.append(
                function_call(tools[tool_id]["class"], result["parameters"])
            )
        elif isinstance(result["parameters"], List):
            for parameters in result["parameters"]:
                call_results.append(function_call(tools[tool_id]["class"], parameters))
        return call_results

    call_results = []
    for i, result in enumerate(tool_results):
        call_results.extend(parse_tool_results(result))
        tool_results[i]["call_results"] = call_results

    call_results_str = "\n\n".join([str(e) for e in call_results if e is not None])
    _LOGGER.info(f"\tCall Results: {call_results_str}")
    return tool_results, call_results_str


def create_tasks(
    task_model: Union[LLM, LMM], question: str, tools: Dict[int, Any], reflections: str
) -> List[Dict]:
    tasks = task_decompose(
        task_model,
        question,
        {k: v["description"] for k, v in tools.items()},
        reflections,
    )
    if tasks is not None:
        task_list = [{"task": task, "id": i + 1} for i, task in enumerate(tasks)]
        task_list = task_topology(task_model, question, task_list)
        try:
            task_list = topological_sort(task_list)
        except Exception:
            _LOGGER.error(f"Failed topological_sort on: {task_list}")
    else:
        task_list = []
    _LOGGER.info(
        f"""Planned tasks:
{tabulate(task_list, headers="keys", tablefmt="mixed_grid")}"""
    )
    return task_list


def self_reflect(
    reflect_model: Union[LLM, LMM],
    question: str,
    tools: Dict[int, Any],
    tool_result: List[Dict],
    final_answer: str,
    image: Optional[Union[str, Path]] = None,
) -> str:
    prompt = VISION_AGENT_REFLECTION.format(
        question=question,
        tools=format_tools(tools),
        tool_results=str(tool_result),
        final_answer=final_answer,
    )
    if issubclass(type(reflect_model), LMM):
        return reflect_model(prompt, image=image)  # type: ignore
    return reflect_model(prompt)


def parse_reflect(reflect: str) -> bool:
    # GPT-4V has a hard time following directions, so make the criteria less strict
    return "finish" in reflect.lower() and len(reflect) < 100


class VisionAgent(Agent):
    r"""Vision Agent is an agent framework that utilizes tools as well as self
    reflection to accomplish tasks, in particular vision tasks. Vision Agent is based
    off of EasyTool https://arxiv.org/abs/2401.06201 and Reflexion
    https://arxiv.org/abs/2303.11366 where it will attempt to complete a task and then
    reflect on whether or not it was able to accomplish the task based off of the plan
    and final results, if not it will redo the task with this newly added reflection.

    Example
    -------
        >>> from vision_agent.agent import VisionAgent
        >>> agent = VisionAgent()
        >>> resp = agent("If red tomatoes cost $5 each and yellow tomatoes cost $2.50 each, what is the total cost of all the tomatoes in the image?", image="tomatoes.jpg")
        >>> print(resp)
        "The total cost is $57.50."
    """

    def __init__(
        self,
        task_model: Optional[Union[LLM, LMM]] = None,
        answer_model: Optional[Union[LLM, LMM]] = None,
        reflect_model: Optional[Union[LLM, LMM]] = None,
        max_retries: int = 2,
        verbose: bool = False,
    ):
        self.task_model = (
            OpenAILLM(json_mode=True) if task_model is None else task_model
        )
        self.answer_model = OpenAILLM() if answer_model is None else answer_model
        self.reflect_model = OpenAILMM() if reflect_model is None else reflect_model
        self.max_retries = max_retries

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
            The result of the vision agent in text.
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

        reflections = ""
        final_answer = ""
        all_tool_results: List[Dict] = []

        for _ in range(self.max_retries):
            task_list = create_tasks(self.task_model, question, self.tools, reflections)

            _LOGGER.info(f"Task Dependency: {task_list}")
            task_depend = {"Original Quesiton": question}
            previous_log = ""
            answers = []
            for task in task_list:
                task_depend[task["id"]] = {"task": task["task"], "answer": "", "call_result": ""}  # type: ignore
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
                    reflections,
                )
                answer = answer_generate(
                    self.answer_model, task_str, call_results, previous_log, reflections
                )

                for tool_result in tool_results:
                    tool_result["answer"] = answer
                all_tool_results.extend(tool_results)

                _LOGGER.info(f"\tAnswer: {answer}")
                answers.append({"task": task_str, "answer": answer})
                task_depend[task["id"]]["answer"] = answer  # type: ignore
                task_depend[task["id"]]["call_result"] = call_results  # type: ignore
            final_answer = answer_summarize(
                self.answer_model, question, answers, reflections
            )

            reflection = self_reflect(
                self.reflect_model,
                question,
                self.tools,
                all_tool_results,
                final_answer,
                image,
            )
            _LOGGER.info(f"\tReflection: {reflection}")
            if parse_reflect(reflection):
                break
            else:
                reflections += reflection

        return final_answer, all_tool_results

    def chat(
        self, chat: List[Dict[str, str]], image: Optional[Union[str, Path]] = None
    ) -> str:
        answer, _ = self.chat_with_workflow(chat, image=image)
        return answer
