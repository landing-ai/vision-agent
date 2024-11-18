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
from vision_agent.agent.vision_agent_planner_prompts_v2 import (
    CATEGORIZE_TOOL_REQUEST,
    FINALIZE_PLAN,
    PICK_TOOL,
    TEST_TOOLS,
    TEST_TOOLS_EXAMPLE1,
    TEST_TOOLS_EXAMPLE2,
)
from vision_agent.lmm import AnthropicLMM
from vision_agent.utils.execute import CodeInterpreterFactory
from vision_agent.utils.image_utils import convert_to_b64
from vision_agent.utils.sim import load_cached_sim

TOOL_FUNCTIONS = {tool.__name__: tool for tool in T.TOOLS}
TOOL_RECOMMENDER = load_cached_sim(T.TOOLS_DF)

_LOGGER = logging.getLogger(__name__)
EXAMPLES = f"\n{TEST_TOOLS_EXAMPLE1}\n{TEST_TOOLS_EXAMPLE2}\n"


def extract_tool_info(
    tool_choice_context: Dict[str, Any]
) -> Tuple[Optional[Callable], str, str, str]:
    tool_thoughts = tool_choice_context.get("thoughts", "")
    tool_docstring = ""
    tool = tool_choice_context.get("best_tool", None)
    if tool in TOOL_FUNCTIONS:
        tool = TOOL_FUNCTIONS[tool]
        tool_docstring = T.TOOLS_INFO[tool.__name__]

    return tool, tool_thoughts, tool_docstring, ""


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
    input and output signatures are.

    Parameters:
        task: str: The task to accomplish.
        images: List[np.ndarray]: The images to use for the task.
        exclude_tools: Optional[List[str]]: A list of tool names to exclude from the
            recommendations. This is helpful if you are calling get_tool_for_task twice
            and do not want the same tool recommended.

    Returns:
        The tool to use for the task is printed to stdout

    Examples
    --------
        >>> get_tool_for_task("Give me an OCR model that can find 'hot chocolate' in the image", [image])
    """
    lmm = AnthropicLMM()

    with (
        tempfile.TemporaryDirectory() as tmpdirname,
        CodeInterpreterFactory.new_instance() as code_interpreter,
    ):
        image_paths = []
        for i, image in enumerate(images[:3]):
            image_path = f"{tmpdirname}/image_{i}.png"
            Image.fromarray(image).save(image_path)
            image_paths.append(image_path)

        query = lmm.generate(CATEGORIZE_TOOL_REQUEST.format(task=task))
        category = extract_tag(query, "category")  # type: ignore
        if category is None:
            category = task
        else:
            category = (
                f"I need models from the {category.strip()} category of tools. {task}"
            )

        tool_docs = TOOL_RECOMMENDER.top_k(category, k=10, thresh=0.2)
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
            examples=EXAMPLES,
            media=str(image_paths),
        )

        response = lmm.generate(prompt, media=image_paths)
        code = extract_tag(response, "code")  # type: ignore
        if code is None:
            raise ValueError(f"Could not extract code from response: {response}")
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
                examples=EXAMPLES,
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
        tool_choice_context_dict = extract_json(tool_choice_context)  # type: ignore

        tool, tool_thoughts, tool_docstring, error_message = extract_tool_info(
            tool_choice_context_dict
        )

        count = 1
        while tool is None and count <= 3:
            prompt = PICK_TOOL.format(
                tool_docs=tool_docs_str,
                user_request=task,
                context=f"<code>\n{code}\n</code>\n<tool_output>\n{tool_output_str}\n</tool_output>",
                previous_attempts=error_message,
            )
            tool_choice_context_dict = extract_json(lmm.generate(prompt, media=image_paths))  # type: ignore
            tool, tool_thoughts, tool_docstring, error_message = extract_tool_info(
                tool_choice_context_dict
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
    finalized_plan = cast(str, lmm.generate(prompt))
    return finalized_plan


def claude35_vqa(prompt: str, medias: List[np.ndarray]) -> None:
    """Asks the Claude-3.5 model a question about the given media and returns an answer.

    Parameters:
        prompt: str: The question to ask the model.
        medias: List[np.ndarray]: The images to ask the question about, it could also
            be frames from a video. You can send up to 5 frames from a video.
    """
    lmm = AnthropicLMM()
    if isinstance(medias, np.ndarray):
        medias = [medias]
    if isinstance(medias, list) and len(medias) > 5:
        medias = medias[:5]
    all_media_b64 = [
        "data:image/png;base64," + convert_to_b64(media) for media in medias
    ]

    response = cast(str, lmm.generate(prompt, media=all_media_b64))
    print(f"[claude35_vqa output]\n{response}\n[end of claude35_vqa output]")


def suggestion(prompt: str, medias: List[np.ndarray]) -> None:
    """Given your problem statement and the images, this will provide you with a
    suggested plan on how to proceed. Always call suggestion when starting to solve
    a problem.

    Parameters:
        prompt: str: The problem statement.
        medias: List[np.ndarray]: The images to use for the problem
    """
    try:
        from .suggestion import suggestion_impl  # type: ignore

        suggestion = suggestion_impl(prompt, medias)
        print(suggestion)
    except ImportError:
        print("")


PLANNER_TOOLS = [
    claude35_vqa,
    suggestion,
    get_tool_for_task,
    T.load_image,
    T.save_image,
    T.extract_frames_and_timestamps,
    T.save_video,
]
PLANNER_DOCSTRING = T.get_tool_documentation(PLANNER_TOOLS)  # type: ignore
