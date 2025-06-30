import inspect
import logging
import math
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import libcst as cst
import numpy as np
from IPython.display import display
from PIL import Image

import vision_agent.tools as T
from vision_agent.agent.vision_agent_planner_prompts_v2 import (
    CATEGORIZE_TOOL_REQUEST,
    FINALIZE_PLAN,
    PICK_TOOL,
    TEST_TOOLS,
    TEST_TOOLS_EXAMPLE1,
    TEST_TOOLS_EXAMPLE2,
)
from vision_agent.configs import Config
from vision_agent.lmm import LMM, AnthropicLMM
from vision_agent.sim import get_tool_recommender
from vision_agent.tools.tools import get_tools, get_tools_info
from vision_agent.utils.agent import DefaultImports, extract_json, extract_tag
from vision_agent.utils.execute import (
    CodeInterpreter,
    CodeInterpreterFactory,
    Execution,
    MimeType,
)
from vision_agent.utils.image_utils import convert_to_b64
from vision_agent.utils.tools_doc import get_tool_documentation


def get_tool_functions() -> Dict[str, Callable]:
    return {tool.__name__: tool for tool in get_tools()}


def get_load_tools_docstring() -> str:
    return get_tool_documentation([T.load_image, T.extract_frames_and_timestamps])


CONFIG = Config()
_LOGGER = logging.getLogger(__name__)
EXAMPLES = f"\n{TEST_TOOLS_EXAMPLE1}\n{TEST_TOOLS_EXAMPLE2}\n"


def format_tool_output(tool_thoughts: str, tool_docstring: str) -> str:
    return_str = "[get_tool_for_task output]\n"
    if tool_thoughts.strip() != "":
        return_str += f"{tool_thoughts}\n\n"
    return_str += (
        f"Tool Documentation:\n{tool_docstring}\n[end of get_tool_for_task output]\n"
    )
    return return_str


def judge_od_results(
    prompt: str,
    image: np.ndarray,
    detections: List[Dict[str, Any]],
) -> str:
    """Given an image and the detections, this function will judge the results and
    return the thoughts on the results.

    Parameters:
        prompt (str): The prompt that was used to generate the detections.
        image (np.ndarray): The image that the detections were made on.
        detections (List[Dict[str, Any]]): The detections made on the image.

    Returns:
        str: The thoughts on the results.
    """

    if not detections:
        return "No detections found in the image."

    od_judge = CONFIG.create_od_judge()
    max_crop_size = (512, 512)

    # Randomly sample up to 10 detections
    num_samples = min(10, len(detections))
    sampled_detections = random.sample(detections, num_samples)
    crops = []
    h, w = image.shape[:2]

    for detection in sampled_detections:
        if "bbox" not in detection:
            continue
        x1, y1, x2, y2 = detection["bbox"]
        crop = image[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]
        if crop.shape[0] > max_crop_size[0] or crop.shape[1] > max_crop_size[1]:
            crop = Image.fromarray(crop)  # type: ignore
            crop.thumbnail(max_crop_size)  # type: ignore
            crop = np.array(crop)
        crops.append("data:image/png;base64," + convert_to_b64(crop))

    sampled_detection_info = [
        {"score": d["score"], "label": d["label"]} for d in sampled_detections
    ]

    prompt = f"""The user is trying to detect '{prompt}' in an image. You are shown 10 images which represent crops of the detected objets. Below is the detection labels and scores:
{sampled_detection_info}

Look over each of the cropped images and corresponding labels and scores. Provide a judgement on whether or not the results are correct. If the results are incorrect you can only suggest a different prompt or a threshold."""

    response = cast(str, od_judge.generate(prompt, media=crops))
    return response


def run_multi_judge(
    tool_chooser: LMM,
    tool_docs_str: str,
    task: str,
    code: str,
    tool_output_str: str,
    image_paths: List[str],
    n_judges: int = 3,
) -> Tuple[Optional[Callable], str, str]:
    error_message = ""
    prompt = PICK_TOOL.format(
        tool_docs=tool_docs_str,
        user_request=task,
        context=f"<code>\n{code}\n</code>\n<tool_output>\n{tool_output_str}\n</tool_output>",
        previous_attempts=error_message,
    )

    def run_judge() -> Tuple[Optional[Callable], str, str]:
        response = tool_chooser.generate(prompt, media=image_paths, temperature=1.0)
        tool_choice_context = extract_tag(response, "json", extract_markdown="json")  # type: ignore
        tool_choice_context_dict = extract_json(tool_choice_context)  # type: ignore
        tool, tool_thoughts, tool_docstring, _ = extract_tool_info(
            tool_choice_context_dict
        )
        return tool, tool_thoughts, tool_docstring

    responses = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_judge) for _ in range(n_judges)]
        for future in as_completed(futures):
            responses.append(future.result())

    responses = [r for r in responses if r[0] is not None]
    counts: Dict[str, int] = {}
    for tool, tool_thoughts, tool_docstring in responses:
        if tool is not None:
            counts[tool.__name__] = counts.get(tool.__name__, 0) + 1
            if counts[tool.__name__] >= math.ceil(n_judges / 2):
                return tool, tool_thoughts, tool_docstring

    if len(responses) == 0:
        return (
            None,
            "No tool could be found, please try again with a different prompt or image",
            "",
        )
    return responses[0]


def extract_tool_info(
    tool_choice_context: Dict[str, Any],
) -> Tuple[Optional[Callable], str, str, str]:
    tool_thoughts = tool_choice_context.get("thoughts", "")
    tool_docstring = ""
    tool = tool_choice_context.get("best_tool", None)
    tools_info = get_tools_info()

    tool_functions = get_tool_functions()
    if tool in tool_functions:
        tool = tool_functions[tool]
        tool_docstring = tools_info[tool.__name__]

    return tool, tool_thoughts, tool_docstring, ""


def replace_box_threshold(code: str, functions: List[str], box_threshold: float) -> str:
    class ReplaceBoxThresholdTransformer(cst.CSTTransformer):
        def leave_Call(
            self, original_node: cst.Call, updated_node: cst.Call
        ) -> cst.Call:
            if (
                isinstance(updated_node.func, cst.Name)
                and updated_node.func.value in functions
            ) or (
                isinstance(updated_node.func, cst.Attribute)
                and updated_node.func.attr.value in functions
            ):
                new_args = []
                found = False
                for arg in updated_node.args:
                    if arg.keyword and arg.keyword.value == "box_threshold":
                        new_arg = arg.with_changes(value=cst.Float(str(box_threshold)))
                        new_args.append(new_arg)
                        found = True
                    else:
                        new_args.append(arg)

                if not found:
                    new_args.append(
                        cst.Arg(
                            keyword=cst.Name("box_threshold"),
                            value=cst.Float(str(box_threshold)),
                            equal=cst.AssignEqual(
                                whitespace_before=cst.SimpleWhitespace(""),
                                whitespace_after=cst.SimpleWhitespace(""),
                            ),
                        )
                    )
                return updated_node.with_changes(args=new_args)
            return updated_node

    tree = cst.parse_module(code)
    transformer = ReplaceBoxThresholdTransformer()
    new_tree = tree.visit(transformer)
    return new_tree.code


def retrieve_tool_docs(lmm: LMM, task: str, exclude_tools: Optional[List[str]]) -> str:
    query = cast(str, lmm.generate(CATEGORIZE_TOOL_REQUEST.format(task=task)))
    categories_str = extract_tag(query, "category")
    if categories_str is None:
        categories = []
    else:
        categories = [e.strip() for e in categories_str.split(",")]

    explanation = query.split("<category>")[0].strip()
    if "</category>" in query:
        explanation += " " + query.split("</category>")[1].strip()
        explanation = explanation.strip()

    sim = get_tool_recommender()

    all_tool_docs = []
    all_tool_doc_names = set()
    exclude_tools = [] if exclude_tools is None else exclude_tools
    for category in categories + [task]:
        tool_docs = sim.top_k(category, k=3, thresh=0.3)

        for tool_doc in tool_docs:
            if (
                tool_doc["name"] not in all_tool_doc_names
                and tool_doc["name"] not in exclude_tools
            ):
                all_tool_docs.append(tool_doc)
                all_tool_doc_names.add(tool_doc["name"])

    tool_docs_str = explanation + "\n\n" + "\n".join([e["doc"] for e in all_tool_docs])
    tool_docs_str += get_load_tools_docstring()
    return tool_docs_str


def run_tool_testing(
    task: str,
    image_paths: List[str],
    lmm: LMM,
    exclude_tools: Optional[List[str]],
    code_interpreter: CodeInterpreter,
    process_code: Callable[[str], str] = lambda x: x,
) -> tuple[str, str, Execution]:
    """Helper function to generate and run tool testing code."""

    tool_docs_str = retrieve_tool_docs(lmm, task, exclude_tools)

    prompt = TEST_TOOLS.format(
        tool_docs=tool_docs_str,
        previous_attempts="",
        user_request=task,
        examples=EXAMPLES,
        media=str(image_paths),
    )

    response = lmm.generate(prompt, media=image_paths)
    code = extract_tag(response, "code", extract_markdown="python")  # type: ignore
    if code is None:
        raise ValueError(f"Could not extract code from response: {response}")

    # If there's a syntax error with the code, process_code can crash. Executing the
    # code and then sending the error to the LLM should correct it.
    try:
        code = process_code(code)
    except Exception as e:
        _LOGGER.error(f"Error processing code: {e}")

    tool_output = code_interpreter.exec_isolation(DefaultImports.prepend_imports(code))
    tool_output_str = tool_output.text(include_results=False).strip()

    count = 1
    while (
        not tool_output.success
        or (len(tool_output.logs.stdout) == 0 and len(tool_output.logs.stderr) == 0)
    ) and count <= 3:
        if tool_output_str.strip() == "":
            tool_output_str = "EMPTY"
        prompt = TEST_TOOLS.format(
            tool_docs=tool_docs_str,
            previous_attempts=f"<code>\n{code}\n</code>\nTOOL OUTPUT\n{tool_output_str}",
            user_request=task,
            examples=EXAMPLES,
            media=str(image_paths),
        )
        response = cast(str, lmm.generate(prompt, media=image_paths))
        code = extract_tag(response, "code", extract_markdown="python")
        if code is None:
            code = response

        try:
            code = process_code(code)
        except Exception as e:
            _LOGGER.error(f"Error processing code: {e}")
        tool_output = code_interpreter.exec_isolation(
            DefaultImports.prepend_imports(code)
        )
        tool_output_str = tool_output.text(include_results=False).strip()
        count += 1

    return code, tool_docs_str, tool_output


def get_tool_for_task(
    task: str,
    images: Union[Dict[str, List[np.ndarray]], List[np.ndarray]],
    exclude_tools: Optional[List[str]] = None,
) -> None:
    """Given a task and one or more images this function will find a tool to accomplish
    the jobs. It prints the tool documentation and thoughts on why it chose the tool.

    It can produce tools for the following types of tasks:
        - Object detection and counting
        - Classification
        - Segmentation
        - OCR
        - VQA
        - Depth and pose estimation
        - Video object tracking
        - Video temporal localization (action recognition)
        - Image inpainting

    Only ask for one type of task at a time, for example a task needing to identify
    text is one OCR task while needing to identify non-text objects is an OD task. Wait
    until the documentation is printed to use the function so you know what the input
    and output signatures are.

    Parameters:
        task (str): The task to accomplish.
        images (Union[Dict[str, List[np.ndarray]], List[np.ndarray]]): The images to use
            for the task. If a key is provided, it is used as the file name.
        exclude_tools (Optional[List[str]]): A list of tool names to exclude from the
            recommendations. This is helpful if you are calling get_tool_for_task twice
            and do not want the same tool recommended.

    Returns:
        None: The function does not return the tool but prints it to stdout.

    Examples
    --------
        >>> get_tool_for_task(
        >>>     "Give me an OCR model that can find 'hot chocolate' in the image",
        >>>     {"image": [image]})
        >>> get_tool_for_task(
        >>>     "I need a tool that can paint a background for this image and maks",
        >>>     {"image": [image], "mask": [mask]})
    """
    tool_tester = CONFIG.create_tool_tester()
    tool_chooser = CONFIG.create_tool_chooser()

    if isinstance(images, list):
        if len(images) > 0 and isinstance(images[0], dict):
            if all(["frame" in image for image in images]):
                images = [image["frame"] for image in images]
            else:
                raise ValueError(
                    f"Expected a list of numpy arrays or a dictionary of strings to lists of numpy arrays, got a list of dictionaries instead: {images}"
                )

    if isinstance(images, list):
        images = {"image": images}

    with (
        tempfile.TemporaryDirectory() as tmpdirname,
        CodeInterpreterFactory.new_instance() as code_interpreter,
    ):
        image_paths = []
        for k in images.keys():
            for i, image in enumerate(images[k]):
                image_path = f"{tmpdirname}/{k}_{i}.png"
                Image.fromarray(image).save(image_path)
                image_paths.append(image_path)

        # run no more than 3 images or else it overloads the LLM
        image_paths = image_paths[:3]
        code, tool_docs_str, tool_output = run_tool_testing(
            task, image_paths, tool_tester, exclude_tools, code_interpreter
        )
        tool_output_str = tool_output.text(include_results=False).strip()

        _, tool_thoughts, tool_docstring = run_multi_judge(
            tool_chooser,
            tool_docs_str,
            task,
            code,
            tool_output_str,
            image_paths,
            n_judges=3,
        )

    print(format_tool_output(tool_thoughts, tool_docstring))


def get_tool_for_task_human_reviewer(
    task: str,
    images: Union[Dict[str, List[np.ndarray]], List[np.ndarray]],
    exclude_tools: Optional[List[str]] = None,
) -> None:
    # NOTE: this will have the same documentation as get_tool_for_task
    tool_tester = CONFIG.create_tool_tester()

    if isinstance(images, list):
        if len(images) > 0 and isinstance(images[0], dict):
            if all(["frame" in image for image in images]):
                images = [image["frame"] for image in images]
            else:
                raise ValueError(
                    f"Expected a list of numpy arrays or a dictionary of strings to lists of numpy arrays, got a list of dictionaries instead: {images}"
                )

    if isinstance(images, list):
        images = {"image": images}

    with (
        tempfile.TemporaryDirectory() as tmpdirname,
        CodeInterpreterFactory.new_instance() as code_interpreter,
    ):
        image_paths = []
        for k in images.keys():
            for i, image in enumerate(images[k]):
                image_path = f"{tmpdirname}/{k}_{i}.png"
                Image.fromarray(image).save(image_path)
                image_paths.append(image_path)

        # run no more than 3 images or else it overloads the LLM
        image_paths = image_paths[:3]

        tools = [
            t.__name__
            for t in get_tools()
            if inspect.signature(t).parameters.get("box_threshold")
        ]

        _, _, tool_output = run_tool_testing(
            task,
            image_paths,
            tool_tester,
            exclude_tools,
            code_interpreter,
            process_code=lambda x: replace_box_threshold(x, tools, 0.05),
        )

        # need to re-display results for the outer notebook to see them
        for result in tool_output.results:
            if "json" in result.formats():
                display({MimeType.APPLICATION_JSON: result.json}, raw=True)


def check_function_call(code: str, function_name: str) -> bool:
    class FunctionCallVisitor(cst.CSTVisitor):
        def __init__(self) -> None:
            self.function_name = function_name
            self.function_called = False

        def visit_Call(self, node: cst.Call) -> None:
            if (
                isinstance(node.func, cst.Name)
                and node.func.value == self.function_name
            ):
                self.function_called = True

    tree = cst.parse_module(code)
    visitor = FunctionCallVisitor()
    tree.visit(visitor)
    return visitor.function_called


def finalize_plan(user_request: str, chain_of_thoughts: str) -> str:
    """Finalizes the plan by taking the user request and the chain of thoughts that
    represent the plan and returns the finalized plan.
    """
    lmm = AnthropicLMM()
    prompt = FINALIZE_PLAN.format(
        user_request=user_request, chain_of_thoughts=chain_of_thoughts
    )
    finalized_plan = cast(str, lmm.generate(prompt))
    return finalized_plan


def vqa(prompt: str, medias: List[np.ndarray]) -> None:
    """Asks the VQA model a question about the given media and returns an answer.

    Parameters:
        prompt: str: The question to ask the model.
        medias: List[np.ndarray]: The images to ask the question about, it could also
            be frames from a video. You can send up to 5 frames from a video.
    """
    vqa = CONFIG.create_vqa()
    if isinstance(medias, np.ndarray):
        medias = [medias]
    if isinstance(medias, list) and len(medias) > 5:
        medias = medias[:5]
    all_media_b64 = [
        "data:image/png;base64," + convert_to_b64(media) for media in medias
    ]

    response = cast(str, vqa.generate(prompt, media=all_media_b64))
    print(f"[vqa output]\n{response}\n[end of vqa output]")


def suggestion(prompt: str, medias: List[np.ndarray]) -> None:
    """Given your problem statement and the images, this will provide you with a
    suggested plan on how to proceed. Always call suggestion when starting to solve
    a problem. 'suggestion' will only print pseudo code for you to execute, it will not
    execute the code for you.

    Parameters:
        prompt: str: The problem statement, provide a detailed description of the
            problem you are trying to solve.
        medias: List[np.ndarray]: The images to use for the problem
    """
    try:
        from .suggestion import suggestion_impl  # type: ignore

        suggestion = suggestion_impl(prompt, medias)
        print(suggestion)
    except ImportError:
        print("")


PLANNER_TOOLS = [
    vqa,
    suggestion,
    get_tool_for_task,
]
PLANNER_DOCSTRING = get_tool_documentation(PLANNER_TOOLS)  # type: ignore
