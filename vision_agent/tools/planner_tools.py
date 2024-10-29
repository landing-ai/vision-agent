import io
import json
import logging
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
from PIL import Image

import vision_agent.tools as T
from vision_agent.agent.agent_utils import (
    DefaultImports,
    extract_code,
    extract_json,
    extract_tag,
)
from vision_agent.agent.plan_repository import (
    CHECK_COLOR,
    LARGE_IMAGE,
    MISSING_GRID_ELEMENTS,
    MISSING_HORIZONTAL_ELEMENTS,
    MISSING_VERTICAL_ELEMENTS,
    SMALL_TEXT,
    SUGGESTIONS,
)
from vision_agent.agent.vision_agent_planner_prompts_v2 import (
    CATEGORIZE_TOOL_REQUEST,
    FINALIZE_PLAN,
    PICK_TOOL,
    TEST_TOOLS,
)
from vision_agent.lmm import AnthropicLMM
from vision_agent.utils.execute import CodeInterpreterFactory
from vision_agent.utils.image_utils import encode_image_bytes
from vision_agent.utils.sim import Sim

TOOL_FUNCTIONS = {tool.__name__: tool for tool in T.TOOLS}
# build this here so we don't have to create it every call to `get_tool_for_task`
TOOL_RECOMMENDER = Sim(df=T.TOOLS_DF, sim_key="doc")
_LOGGER = logging.getLogger(__name__)
TOOL_LIST_PRIORS = {
    "owl_v2_image": 0.6,
    "owl_v2_video": 0.6,
    "ocr": 0.8,
    "clip": 0.6,
    "vit_image_classification": 0.5,
    "vit_nsfw_classification": 0.5,
    "countgd_counting": 0.9,
    "florence2_ocr": 0.8,
    "florence2_sam2_image": 0.6,
    "florence2_sam2_video_tracking": 0.8,
    "florence2_phrase_grounding": 0.6,
    "ixc25_image_vqa": 0.6,
    "ixc25_video_vqa": 0.6,
    "detr_segmentation": 0.5,
    "depth_anything_v2": 0.6,
    "generate_pose_image": 0.5,
}


def _get_b64_images(medias: List[np.ndarray]) -> List[str]:
    all_media_b64 = []
    for media in medias:
        buffer = io.BytesIO()
        Image.fromarray(media).save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_b64 = "data:image/png;base64," + encode_image_bytes(image_bytes)
        all_media_b64.append(image_b64)
    return all_media_b64


def extract_tool_info(
    tool_choice_context: Dict[str, Any]
) -> Tuple[Optional[Callable], str, str, str]:
    error_message = ""
    tool_docstring = "No tool was found."
    tool_thoughts = ""
    tool_posteriors = {}
    for tool_name in tool_choice_context:
        if tool_name in TOOL_LIST_PRIORS:
            prior = TOOL_LIST_PRIORS[tool_name]
            try:
                score = float(tool_choice_context[tool_name]) / 10
            except ValueError:
                score = 0.5
            posterior = prior * score
            tool_posteriors[tool_name] = posterior

    if len(tool_posteriors) == 0:
        return None, "", "", "No tool was found."

    best_tool = max(tool_posteriors, key=tool_posteriors.get)
    tool_docstring = T.TOOLS_INFO[best_tool]
    tool = TOOL_FUNCTIONS[best_tool]

    if "thoughts" in tool_choice_context:
        tool_thoughts = tool_choice_context["thoughts"]
    return tool, tool_thoughts, tool_docstring, error_message


def get_tool_for_task(
    task: str, images: List[np.ndarray], exclude_tools: Optional[List[str]] = None
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

    Wait until the documentation is printed to use the function so you know what the
    input and output signatures are. For text detection and extraction tasks, provide
    the text you want to extract in the task string to help the model find the right
    tool.

    Parameters:
        task: str: The task to accomplish.
        images: List[np.ndarray]: The images to use for the task.
        exclude_tools: Optional[List[str]]: A list of tool names to exclude from the
            recommendations. This is helpful if you are calling get_tool_for_task twice
            and do not want the same tool recommended.

    Returns:
        None: The tool to use for the task is printed to stdout
    """
    lmm = AnthropicLMM()

    with (
        tempfile.TemporaryDirectory() as tmpdirname,
        CodeInterpreterFactory.new_instance() as code_interpreter,
    ):
        image_paths = []
        for i, image in enumerate(images):
            image_path = f"{tmpdirname}/image_{i}.png"
            Image.fromarray(image).save(image_path)
            image_paths.append(image_path)

        query = lmm.generate(CATEGORIZE_TOOL_REQUEST.format(task=task))  # type: ignore
        category = extract_tag(query, "category")  # type: ignore
        if category is None:
            category = task

        tool_docs = TOOL_RECOMMENDER.top_k(category, k=5, thresh=0.2)
        if exclude_tools is not None and len(exclude_tools) > 0:
            cleaned_tool_docs = []
            for tool_doc in tool_docs:
                if not tool_doc["name"] in exclude_tools:
                    cleaned_tool_docs.append(tool_doc)
            tool_docs = cleaned_tool_docs
        tool_docs_str = "\n".join([e["doc"] for e in tool_docs])

        prompt = TEST_TOOLS.format(
            tool_docs=tool_docs_str,
            previous_attempts="",
            user_request=task,
            media=str(image_paths),
        )

        response = lmm.generate(prompt, media=image_paths)
        code = extract_tag(response, "code")  # type: ignore
        tool_output = code_interpreter.exec_isolation(
            DefaultImports.prepend_imports(code)
        )
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
                media=str(image_paths),
            )
            code = extract_code(lmm.generate(prompt, media=image_paths))  # type: ignore
            tool_output = code_interpreter.exec_isolation(
                DefaultImports.prepend_imports(code)
            )
            tool_output_str = tool_output.text(include_results=False).strip()

        error_message = ""
        prompt = PICK_TOOL.format(
            tool_docs=tool_docs_str,
            user_request=task,
            context=f"<code>\n{code}\n</code>\n<tool_output>\n{tool_output_str}\n</tool_output>",
            previous_attempts=error_message,
        )

        response = lmm.generate(prompt, media=image_paths)
        tool_choice_context = extract_tag(response, "json")  # type: ignore
        tool_choice_context = extract_json(tool_choice_context)  # type: ignore

        tool, tool_thoughts, tool_docstring, error_message = extract_tool_info(
            tool_choice_context
        )

        count = 1
        while tool is None and count <= 3:
            prompt = PICK_TOOL.format(
                tool_docs=tool_docs_str,
                user_request=task,
                context=f"<code>\n{code}\n</code>\n<tool_output>\n{tool_output_str}\n</tool_output>",
                previous_attempts=error_message,
            )
            tool_choice_context = extract_json(lmm.generate(prompt, media=image_paths))  # type: ignore
            tool, tool_thoughts, tool_docstring, error_message = extract_tool_info(
                tool_choice_context
            )
        try:
            shutil.rmtree(tmpdirname)
        except Exception as e:
            _LOGGER.error(f"Error removing temp directory: {e}")

    print(
        f"[get_tool_for_task output]\n{tool_thoughts}\n\nTool Documentation:\n{tool_docstring}\n[end of get_tool_for_task output]\n"
    )


def finalize_plan(user_request: str, chain_of_thoughts: str) -> str:
    """Finalizes the plan by taking the user request and the chain of thoughts that
    represent the plan and returns the finalized plan.
    """
    lmm = AnthropicLMM()
    prompt = FINALIZE_PLAN.format(
        user_request=user_request, chain_of_thoughts=chain_of_thoughts
    )
    finalized_plan = cast(str, lmm.generate(prompt))  # type: ignore
    return finalized_plan


def claude35_vqa(prompt: str, medias: List[np.ndarray]) -> None:
    """Asks the Claude-3.5 model a question about the given media and returns an answer.

    Parameters:
        prompt: str: The question to ask the model.
        medias: List[np.ndarray]: The images to ask the question about.
    """
    lmm = AnthropicLMM()
    if isinstance(medias, np.ndarray):
        medias = [medias]
    all_media_b64 = _get_b64_images(medias)

    response = cast(str, lmm.generate(prompt, media=all_media_b64))  # type: ignore
    print(f"[claude35_vqa output]\n{response}\n[end of claude35_vqa output]")


def suggestion(prompt: str, medias: List[np.ndarray]) -> None:
    """Given your problem statement and the images, this will provide you with a
    suggested plan on how to proceed. Always call suggestion when starting to solve
    a problem.

    Parameters:
        prompt: str: The problem statement.
        medias: List[np.ndarray]: The images to use for the problem
    """
    lmm = AnthropicLMM()
    if isinstance(medias, np.ndarray):
        medias = [medias]
    all_media_b64 = _get_b64_images(medias)
    image_sizes = [media.shape for media in medias]
    image_size_info = (
        " The original image sizes were "
        + str(image_sizes)
        + ", I have resized them to 768x768, if my resize is much smaller than the original image size I may have missed some details."
    )
    prompt = SUGGESTIONS.format(user_request=prompt, image_size_info=image_size_info)
    response = cast(str, lmm.generate(prompt, media=all_media_b64))  # type: ignore
    json_str = extract_tag(response, "json")

    try:
        output = extract_json(json_str)
    except json.JSONDecodeError as e:
        _LOGGER.error(f"Error decoding JSON: {e}")
        output = {"reason": "No strategies found", "categories": []}
    reason = output["reason"]
    categories = set(output["categories"])

    suggestion = ""
    i = 0
    for suggestion_and_cat in [
        LARGE_IMAGE,
        SMALL_TEXT,
        CHECK_COLOR,
        MISSING_GRID_ELEMENTS,
        MISSING_HORIZONTAL_ELEMENTS,
        MISSING_VERTICAL_ELEMENTS,
    ]:
        if len(categories & suggestion_and_cat[1]) > 0:
            suggestion += (
                f"\n[suggestion {i}]\n"
                + suggestion_and_cat[0]
                + f"\n[end of suggestion {i}]"
            )
            i += 1

    response = f"[suggestions]\n{reason}\n{suggestion}\n[end of suggestions]"
    print(response)


PLANNER_TOOLS = [
    claude35_vqa,
    suggestion,
    get_tool_for_task,
    T.load_image,
    T.save_image,
    T.extract_frames_and_timestamps,
    T.save_video,
]
PLANNER_DOCSTRING = T.get_tool_documentation(PLANNER_TOOLS)
