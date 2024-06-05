import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image
from tabulate import tabulate

from vision_agent.agent.agent import Agent
from vision_agent.agent.easytool_prompts import (
    ANSWER_GENERATE,
    ANSWER_SUMMARIZE,
    CHOOSE_PARAMETER,
    CHOOSE_TOOL,
    TASK_DECOMPOSE,
    TASK_TOPOLOGY,
)
from vision_agent.agent.easytool_v2_prompts import (
    ANSWER_GENERATE_DEPENDS,
    ANSWER_SUMMARIZE_DEPENDS,
    CHOOSE_PARAMETER_DEPENDS,
    CHOOSE_TOOL_DEPENDS,
    TASK_DECOMPOSE_DEPENDS,
    VISION_AGENT_REFLECTION,
)
from vision_agent.llm import LLM, OpenAILLM
from vision_agent.lmm import LMM, OpenAILMM
from vision_agent.tools.easytool_tools import TOOLS
from vision_agent.utils.image_utils import (
    convert_to_b64,
    overlay_bboxes,
    overlay_heat_map,
    overlay_masks,
)

logging.basicConfig(stream=sys.stdout)
_LOGGER = logging.getLogger(__name__)
_MAX_TABULATE_COL_WIDTH = 80


def parse_json(s: str) -> Any:
    s = (
        s.replace(": True", ": true")
        .replace(": False", ": false")
        .replace(":True", ": true")
        .replace(":False", ": false")
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


def format_tool_usage(tools: Dict[int, Any], tool_result: List[Dict]) -> str:
    usage = []
    name_to_usage = {v["name"]: v["usage"] for v in tools.values()}
    for tool_res in tool_result:
        if "tool_name" in tool_res:
            usage.append((tool_res["tool_name"], name_to_usage[tool_res["tool_name"]]))

    usage_str = ""
    for tool_name, tool_usage in usage:
        usage_str += f"{tool_name} - {tool_usage}\n"
    return usage_str


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


def self_reflect(
    reflect_model: Union[LLM, LMM],
    question: str,
    tools: Dict[int, Any],
    tool_result: List[Dict],
    final_answer: str,
    images: Optional[Sequence[Union[str, Path]]] = None,
) -> str:
    prompt = VISION_AGENT_REFLECTION.format(
        question=question,
        tools=format_tools({k: v["description"] for k, v in tools.items()}),
        tool_usage=format_tool_usage(tools, tool_result),
        tool_results=str(tool_result),
        final_answer=final_answer,
    )
    if (
        issubclass(type(reflect_model), LMM)
        and images is not None
        and all([Path(image).suffix in [".jpg", ".jpeg", ".png"] for image in images])
    ):
        return reflect_model(prompt, images=images)  # type: ignore
    return reflect_model(prompt)


def parse_reflect(reflect: str) -> Any:
    reflect = reflect.strip()
    try:
        return parse_json(reflect)
    except Exception:
        _LOGGER.error(f"Failed parse json reflection: {reflect}")
    # LMMs have a hard time following directions, so make the criteria less strict
    finish = (
        "finish" in reflect.lower() and len(reflect) < 100
    ) or "finish" in reflect.lower()[-10:]
    return {"Finish": finish, "Reflection": reflect}


def _handle_extract_frames(
    image_to_data: Dict[str, Dict], tool_result: Dict
) -> Dict[str, Dict]:
    image_to_data = image_to_data.copy()
    # handle extract_frames_ case, useful if it extracts frames but doesn't do
    # any following processing
    for video_file_output in tool_result["call_results"]:
        # When the video tool is run with wrong parameters, exit the loop
        if not isinstance(video_file_output, tuple) or len(video_file_output) < 2:
            break
        for frame, _ in video_file_output:
            image = frame
            if image not in image_to_data:
                image_to_data[image] = {
                    "bboxes": [],
                    "masks": [],
                    "heat_map": [],
                    "labels": [],
                    "scores": [],
                }
    return image_to_data


def _handle_viz_tools(
    image_to_data: Dict[str, Dict], tool_result: Dict
) -> Dict[str, Dict]:
    image_to_data = image_to_data.copy()

    # handle grounding_sam_ and grounding_dino_
    parameters = tool_result["parameters"]
    # parameters can either be a dictionary or list, parameters can also be malformed
    # becaus the LLM builds them
    if isinstance(parameters, dict):
        if "image" not in parameters:
            return image_to_data
        parameters = [parameters]
    elif isinstance(tool_result["parameters"], list):
        if len(tool_result["parameters"]) < 1 or (
            "image" not in tool_result["parameters"][0]
        ):
            return image_to_data

    for param, call_result in zip(parameters, tool_result["call_results"]):
        # Calls can fail, so we need to check if the call was successful. It can either:
        # 1. return a str or some error that's not a dictionary
        # 2. return a dictionary but not have the necessary keys

        if not isinstance(call_result, dict) or (
            "bboxes" not in call_result
            and "mask" not in call_result
            and "heat_map" not in call_result
        ):
            return image_to_data

        # if the call was successful, then we can add the image data
        image = param["image"]
        if image not in image_to_data:
            image_to_data[image] = {
                "bboxes": [],
                "masks": [],
                "heat_map": [],
                "labels": [],
                "scores": [],
            }

        image_to_data[image]["bboxes"].extend(call_result.get("bboxes", []))
        image_to_data[image]["labels"].extend(call_result.get("labels", []))
        image_to_data[image]["scores"].extend(call_result.get("scores", []))
        image_to_data[image]["masks"].extend(call_result.get("masks", []))
        # only single heatmap is returned
        if "heat_map" in call_result:
            image_to_data[image]["heat_map"].append(call_result["heat_map"])
        if "mask_shape" in call_result:
            image_to_data[image]["mask_shape"] = call_result["mask_shape"]

    return image_to_data


def sample_n_evenly_spaced(lst: Sequence, n: int) -> Sequence:
    if n <= 0:
        return []
    elif len(lst) == 0:
        return []
    elif n == 1:
        return [lst[0]]
    elif n >= len(lst):
        return lst

    spacing = (len(lst) - 1) / (n - 1)
    return [lst[round(spacing * i)] for i in range(n)]


def visualize_result(all_tool_results: List[Dict]) -> Sequence[Union[str, Path]]:
    image_to_data: Dict[str, Dict] = {}
    for tool_result in all_tool_results:
        # only handle bbox/mask tools or frame extraction
        if tool_result["tool_name"] not in [
            "grounding_sam_",
            "grounding_dino_",
            "extract_frames_",
            "dinov_",
            "zero_shot_counting_",
            "visual_prompt_counting_",
            "ocr_",
        ]:
            continue

        if tool_result["tool_name"] == "extract_frames_":
            image_to_data = _handle_extract_frames(image_to_data, tool_result)
        else:
            image_to_data = _handle_viz_tools(image_to_data, tool_result)

    visualized_images = []
    for image_str in image_to_data:
        image_path = Path(image_str)
        image_data = image_to_data[image_str]
        if "_counting_" in tool_result["tool_name"]:
            image = overlay_heat_map(image_path, image_data)
        else:
            image = overlay_masks(image_path, image_data)
            image = overlay_bboxes(image, image_data)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name)
            visualized_images.append(f.name)
    return visualized_images


class EasyToolV2(Agent):
    """EasyToolV2 is an agent framework that utilizes tools as well as self reflection
    to accomplish tasks, in particular vision tasks. EasyToolV2 is based off of EasyTool
    https://arxiv.org/abs/2401.06201 and Reflexion https://arxiv.org/abs/2303.11366
    where it will attempt to complete a task and then reflect on whether or not it was
    able to accomplish the task based off of the plan and final results, if not it will
    redo the task with this newly added reflection.

    Example
    -------
        >>> from vision_agent.agent import EasyToolV2
        >>> agent = EasyToolV2()
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
        report_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """EasyToolV2 constructor.

        Parameters:
            task_model: the model to use for task decomposition.
            answer_model: the model to use for reasoning and concluding the answer.
            reflect_model: the model to use for self reflection.
            max_retries: maximum number of retries to attempt to complete the task.
            verbose: whether to print more logs.
            report_progress_callback: a callback to report the progress of the agent.
                This is useful for streaming logs in a web application where multiple
                EasyToolV2 instances are running in parallel. This callback ensures
                that the progress are not mixed up.
        """
        self.task_model = (
            OpenAILLM(model_name="gpt-4-turbo", json_mode=True, temperature=0.0)
            if task_model is None
            else task_model
        )
        self.answer_model = (
            OpenAILLM(model_name="gpt-4-turbo", temperature=0.0)
            if answer_model is None
            else answer_model
        )
        self.reflect_model = (
            OpenAILMM(model_name="gpt-4-turbo", json_mode=True, temperature=0.0)
            if reflect_model is None
            else reflect_model
        )
        self.max_retries = max_retries
        self.tools = TOOLS
        self.report_progress_callback = report_progress_callback
        if verbose:
            _LOGGER.setLevel(logging.INFO)

    def __call__(
        self,
        input: Union[List[Dict[str, str]], str],
        media: Optional[Union[str, Path]] = None,
        reference_data: Optional[Dict[str, str]] = None,
        visualize_output: Optional[bool] = False,
        self_reflection: Optional[bool] = True,
    ) -> str:
        """Invoke the vision agent.

        Parameters:
            input: A conversation in the format of
                [{"role": "user", "content": "describe your task here..."}] or a string
                containing just the content.
            media: The input media referenced in the chat parameter.
            reference_data: A dictionary containing the reference image, mask or bounding
                box in the format of:
                {"image": "image.jpg", "mask": "mask.jpg", "bbox": [0.1, 0.2, 0.1, 0.2]}
                where the bounding box coordinates are normalized.
            visualize_output: Whether to visualize the output.
            self_reflection: boolean to enable and disable self reflection.

        Returns:
            The result of the vision agent in text.
        """
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        return self.chat(
            input,
            media=media,
            visualize_output=visualize_output,
            reference_data=reference_data,
            self_reflection=self_reflection,
        )

    def log_progress(self, data: Dict[str, Any]) -> None:
        _LOGGER.info(data)
        if self.report_progress_callback:
            self.report_progress_callback(data)

    def _report_visualization_via_callback(
        self, images: Sequence[Union[str, Path]]
    ) -> None:
        """This is intended for streaming the visualization images via the callback to the client side."""
        if self.report_progress_callback:
            self.report_progress_callback({"log": "<VIZ>"})
            if images:
                for img in images:
                    self.report_progress_callback(
                        {"log": f"<IMG>base:64{convert_to_b64(img)}</IMG>"}
                    )
            self.report_progress_callback({"log": "</VIZ>"})

    def chat_with_workflow(
        self,
        chat: List[Dict[str, str]],
        media: Optional[Union[str, Path]] = None,
        reference_data: Optional[Dict[str, str]] = None,
        visualize_output: Optional[bool] = False,
        self_reflection: Optional[bool] = True,
    ) -> Tuple[str, List[Dict]]:
        """Chat with EasyToolV2 and return the final answer and all tool results.

        Parameters:
            chat: A conversation in the format of
                [{"role": "user", "content": "describe your task here..."}].
            media: The media image referenced in the chat parameter.
            reference_data: A dictionary containing the reference image, mask or bounding
                box in the format of:
                {"image": "image.jpg", "mask": "mask.jpg", "bbox": [0.1, 0.2, 0.1, 0.2]}
                where the bounding box coordinates are normalized.
            visualize_output: Whether to visualize the output.
            self_reflection: boolean to enable and disable self reflection.

        Returns:
            Tuple[str, List[Dict]]: A tuple where the first item is the final answer
                and the second item is a list of all the tool results.
        """
        if len(chat) == 0:
            raise ValueError("Input cannot be empty.")

        question = chat[0]["content"]
        if media:
            question += f" Image name: {media}"
        if reference_data:
            question += (
                f" Reference image: {reference_data['image']}"
                if "image" in reference_data
                else ""
            )
            question += (
                f" Reference mask: {reference_data['mask']}"
                if "mask" in reference_data
                else ""
            )
            question += (
                f" Reference bbox: {reference_data['bbox']}"
                if "bbox" in reference_data
                else ""
            )

        reflections = ""
        final_answer = ""
        all_tool_results: List[Dict] = []

        for _ in range(self.max_retries):
            task_list = self.create_tasks(
                self.task_model, question, self.tools, reflections
            )

            task_depend = {"Original Question": question}
            previous_log = ""
            answers = []
            for task in task_list:
                task_depend[task["id"]] = {"task": task["task"], "answer": "", "call_result": ""}  # type: ignore
            all_tool_results = []

            for task in task_list:
                task_str = task["task"]
                previous_log = str(task_depend)
                tool_results, call_results = self.retrieval(
                    self.task_model,
                    task_str,
                    self.tools,
                    previous_log,
                    reflections,
                )
                answer = answer_generate(
                    self.answer_model, task_str, call_results, previous_log, reflections
                )

                tool_results["answer"] = answer
                all_tool_results.append(tool_results)

                self.log_progress({"log": f"\tCall Result: {call_results}"})
                self.log_progress({"log": f"\tAnswer: {answer}"})
                answers.append({"task": task_str, "answer": answer})
                task_depend[task["id"]]["answer"] = answer  # type: ignore
                task_depend[task["id"]]["call_result"] = call_results  # type: ignore
            final_answer = answer_summarize(
                self.answer_model, question, answers, reflections
            )
            visualized_output = visualize_result(all_tool_results)
            all_tool_results.append({"visualized_output": visualized_output})
            if len(visualized_output) > 0:
                reflection_images = sample_n_evenly_spaced(visualized_output, 3)
            elif media is not None:
                reflection_images = [media]
            else:
                reflection_images = None

            if self_reflection:
                reflection = self_reflect(
                    self.reflect_model,
                    question,
                    self.tools,
                    all_tool_results,
                    final_answer,
                    reflection_images,
                )
                self.log_progress({"log": f"Reflection: {reflection}"})
                parsed_reflection = parse_reflect(reflection)
                if parsed_reflection["Finish"]:
                    break
                else:
                    reflections += "\n" + parsed_reflection["Reflection"]
            else:
                self.log_progress(
                    {"log": "Self Reflection skipped based on user request."}
                )
                break
        # '<ANSWER>' is a symbol to indicate the end of the chat, which is useful for streaming logs.
        self.log_progress(
            {
                "log": f"EasyToolV2 has concluded this chat. <ANSWER>{final_answer}</ANSWER>"
            }
        )

        if visualize_output:
            viz_images: Sequence[Union[str, Path]] = all_tool_results[-1][
                "visualized_output"
            ]
            self._report_visualization_via_callback(viz_images)
            for img in viz_images:
                Image.open(img).show()

        return final_answer, all_tool_results

    def chat(
        self,
        chat: List[Dict[str, str]],
        media: Optional[Union[str, Path]] = None,
        reference_data: Optional[Dict[str, str]] = None,
        visualize_output: Optional[bool] = False,
        self_reflection: Optional[bool] = True,
    ) -> str:
        answer, _ = self.chat_with_workflow(
            chat,
            media=media,
            visualize_output=visualize_output,
            reference_data=reference_data,
            self_reflection=self_reflection,
        )
        return answer

    def retrieval(
        self,
        model: Union[LLM, LMM, Agent],
        question: str,
        tools: Dict[int, Any],
        previous_log: str,
        reflections: str,
    ) -> Tuple[Dict, str]:
        tool_id = choose_tool(
            model,
            question,
            {k: v["description"] for k, v in tools.items()},
            reflections,
        )
        if tool_id is None:
            return {}, ""

        tool_instructions = tools[tool_id]
        tool_usage = tool_instructions["usage"]
        tool_name = tool_instructions["name"]

        parameters = choose_parameter(
            model, question, tool_usage, previous_log, reflections
        )
        if parameters is None:
            return {}, ""
        tool_results = {
            "task": question,
            "tool_name": tool_name,
            "parameters": parameters,
        }

        self.log_progress(
            {
                "log": f"""Going to run the following tool(s) in sequence:
{tabulate(tabular_data=[tool_results], headers="keys", tablefmt="mixed_grid", maxcolwidths=_MAX_TABULATE_COL_WIDTH)}"""
            }
        )

        def parse_tool_results(result: Dict[str, Union[Dict, List]]) -> Any:
            call_results: List[Any] = []
            if isinstance(result["parameters"], Dict):
                call_results.append(
                    function_call(tools[tool_id]["class"], result["parameters"])
                )
            elif isinstance(result["parameters"], List):
                for parameters in result["parameters"]:
                    call_results.append(
                        function_call(tools[tool_id]["class"], parameters)
                    )
            return call_results

        call_results = parse_tool_results(tool_results)
        tool_results["call_results"] = call_results

        call_results_str = str(call_results)
        return tool_results, call_results_str

    def create_tasks(
        self,
        task_model: Union[LLM, LMM],
        question: str,
        tools: Dict[int, Any],
        reflections: str,
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
        self.log_progress(
            {
                "log": "Planned tasks:",
                "plan": task_list,
            }
        )
        return task_list
