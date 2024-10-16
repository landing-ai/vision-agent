import io
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
from PIL import Image

import vision_agent.tools as T
from vision_agent.agent.agent_utils import DefaultImports, extract_code, extract_json
from vision_agent.agent.vision_agent_planner_prompts import (
    FINALIZE_PLAN,
    PICK_TOOL2,
    TEST_TOOLS2,
)
from vision_agent.lmm import AnthropicLMM
from vision_agent.utils.execute import CodeInterpreterFactory
from vision_agent.utils.image_utils import encode_image_bytes
from vision_agent.utils.sim import Sim

TOOL_FUNCTIONS = {tool.__name__: tool for tool in T.TOOLS}
# build this here so we don't have to create it every call to `get_tool_for_task`
TOOL_RECOMMENDER = Sim(df=T.TOOLS_DF, sim_key="desc")


def claude35_vqa(prompt: str, medias: List[np.ndarray]) -> str:
    """Asks the Claude-3.5 model a question about the given media and returns the answer.

    Parameters:
        prompt: str: The question to ask the model.
        medias: List[np.ndarray]: The images to ask the question about.

    Returns:
        str: The answer to the question.
    """
    lmm = AnthropicLMM()
    all_media_b64 = []
    for media in medias:
        buffer = io.BytesIO()
        Image.fromarray(media).save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_b64 = "data:image/png;base64," + encode_image_bytes(image_bytes)
        all_media_b64.append(image_b64)
    response = cast(str, lmm.generate(prompt, media=all_media_b64))  # type: ignore
    image_sizes = [media.shape for media in medias]
    response += (
        " The original image sizes were "
        + str(image_sizes)
        + ", I have resized them to 768x768, if my resize is much smaller than the original image size I may have missed some details."
    )
    print(f'[claude35_vqa output]: "{response}"')
    return response


def extract_tool_info(
    tool_choice_context: Dict[str, Any]
) -> Tuple[Optional[Callable], str, str, str]:
    error_message = ""
    tool_docstring = "No tool was found."
    tool_thoughts = ""
    tool = None
    if "tool" in tool_choice_context and tool_choice_context["tool"] in T.TOOLS_INFO:
        tool_docstring = T.TOOLS_INFO[tool_choice_context["tool"]]
        tool = TOOL_FUNCTIONS[tool_choice_context["tool"]]
    else:
        error_message = f"Was not able to locate the tool you suggested in the tools list.\n{str(tool_choice_context)}"

    if "thoughts" in tool_choice_context:
        tool_thoughts = tool_choice_context["thoughts"]
    return tool, tool_thoughts, tool_docstring, error_message


def get_tool_for_task(task: str, images: List[np.ndarray]) -> None:
    """Given a task and one or more images this function will find a tool to accomplish
    the jobs. It prints the tool documentation and thoughts on why it chose the tool.

    Wait until the documentation is printed to use the function so
    you know what the input and output signatures are.

    Parameters:
        task: str: The task to accomplish.
        images: List[np.ndarray]: The images to use for the task.

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

        tool_docs = TOOL_RECOMMENDER.top_k(task, k=5, thresh=0.3)
        tool_docs_str = "\n".join([e["doc"] for e in tool_docs])

        prompt = TEST_TOOLS2.format(
            tool_docs=tool_docs_str,
            previous_attempts="",
            user_request=task,
            media=str(image_paths),
        )

        response = lmm.generate(prompt, media=image_paths)
        code = extract_code(response)  # type: ignore
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
            prompt = TEST_TOOLS2.format(
                tool_docs=tool_docs_str,
                previous_attempts="```python\n"
                + code
                + "```\nTOOL OUTPUT\n"
                + tool_output_str,
                user_request=task,
                media=str(image_paths),
            )
            code = extract_code(lmm.generate(prompt, media=image_paths))  # type: ignore
            tool_output = code_interpreter.exec_isolation(
                DefaultImports.prepend_imports(code)
            )
            tool_output_str = tool_output.text(include_results=False).strip()

        error_message = ""
        prompt = PICK_TOOL2.format(
            tool_docs=tool_docs_str,
            user_request=task,
            context="```python\n" + code + "\n```\n" + tool_output_str,
            previous_attempts=error_message,
        )

        tool_choice_context = extract_json(lmm.generate(prompt, media=image_paths))  # type: ignore

        tool, tool_thoughts, tool_docstring, error_message = extract_tool_info(
            tool_choice_context
        )

        count = 1
        while tool is None and count <= 3:
            prompt = PICK_TOOL2.format(
                tool_docs=tool_docs_str,
                user_request=task,
                context="```python\n" + code + "\n```\n" + tool_output_str,
                previous_attempts=error_message,
            )
            tool_choice_context = extract_json(lmm.generate(prompt, media=image_paths))  # type: ignore
            tool, tool_thoughts, tool_docstring, error_message = extract_tool_info(
                tool_choice_context
            )

    print(
        f"[get_tool_for_task output]: {tool_thoughts}\n\nTool Documentation:\n{tool_docstring}"
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


PLANNER_TOOLS = [claude35_vqa, get_tool_for_task]
PLANNER_DOCSTRING = T.get_tool_documentation(PLANNER_TOOLS)
