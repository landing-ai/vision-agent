import io
import tempfile
from typing import Callable, List, Optional

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
    return lmm.generate(prompt, media=all_media_b64)  # type: ignore


def get_tool_for_task(task: str, images: List[np.ndarray]) -> Optional[Callable]:
    """Given a task and one or more images this function tests and finds tools to
    accomplish that task. It will return thoughts on the tool choice and documentation
    for the tool.

    Parameters:
        task: str: The task to accomplish.
        images: List[np.ndarray]: The images to use for the task.

    Returns:
        Optional[Callable]: The tool that can accomplish the task.
    """
    lmm = AnthropicLMM()
    tool_recommender = Sim(df=T.TOOLS_DF, sim_key="desc")

    with (
        tempfile.TemporaryDirectory() as tmpdirname,
        CodeInterpreterFactory.new_instance() as code_interpreter,
    ):
        image_paths = []
        for i, image in enumerate(images):
            image_path = f"{tmpdirname}/image_{i}.png"
            Image.fromarray(image).save(image_path)
            image_paths.append(image_path)

        tool_docs = tool_recommender.top_k(task, k=5, thresh=0.3)

        prompt = TEST_TOOLS2.format(
            tool_docs=tool_docs,
            previous_attempts="",
            user_request=task,
            media=str(image_paths),
        )

        count = 0
        code = extract_code(lmm.generate(prompt, media=image_paths))  # type: ignore
        tool_output = code_interpreter.exec_isolation(
            DefaultImports.prepend_imports(code)
        )
        tool_output_str = tool_output.text(include_results=False).strip()

        while (
            not tool_output.success
            or (len(tool_output.logs.stdout) == 0 and len(tool_output.logs.stderr) == 0)
        ) and count < 3:
            if tool_output_str.strip() == "":
                tool_output_str = "EMPTY"
            prompt = TEST_TOOLS2.format(
                tool_docs=tool_docs,
                previous_attempts=tool_output_str,
                user_request=task,
                media=str(image_paths),
            )
            code = extract_code(lmm.generate(prompt, media=image_paths))  # type: ignore
            tool_output = code_interpreter.exec_isolation(
                DefaultImports.prepend_imports(code)
            )
            tool_output_str = tool_output.text(include_results=False).strip()

    prompt = PICK_TOOL2.format(
        user_request=task,
        context="```python\n" + code + "\n```\n" + tool_output_str,
    )

    tool_choice = extract_json(lmm.generate(prompt, media=image_paths))  # type: ignore
    tool_thoughts = ""
    tool_docstring = "No tool was found."
    tool = None
    if tool_choice in T.TOOLS_INFO:
        tool_docstring = T.TOOLS_INFO[tool_choice]
        tool = TOOL_FUNCTIONS[tool_choice]
    if "thoughts" in tool_choice:
        tool_thoughts = tool_choice["thoughts"]

    print(f"{tool_thoughts}\nTool Documentation:\n{tool_docstring}")
    return tool


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


PLANNER_DOCSTRING = T.get_tool_documentation([claude35_vqa, get_tool_for_task])
