import inspect
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import libcst as cst
import numpy as np
from IPython.display import display
from PIL import Image

import vision_agent.tools as T
from vision_agent.agent.agent_utils import DefaultImports, extract_json, extract_tag
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
from vision_agent.utils.execute import (
    CodeInterpreter,
    CodeInterpreterFactory,
    Execution,
    MimeType,
)
from vision_agent.utils.image_utils import convert_to_b64
from vision_agent.utils.sim import get_tool_recommender

TOOL_FUNCTIONS = {tool.__name__: tool for tool in T.TOOLS}
LOAD_TOOLS_DOCSTRING = T.get_tool_documentation(
    [T.load_image, T.extract_frames_and_timestamps]
)

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


def run_multi_judge(
    tool_chooser: LMM,
    tool_docs_str: str,
    task: str,
    code: str,
    tool_output_str: str,
    image_paths: List[str],
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
        tool_choice_context = extract_tag(response, "json")  # type: ignore
        tool_choice_context_dict = extract_json(tool_choice_context)  # type: ignore
        tool, tool_thoughts, tool_docstring, _ = extract_tool_info(
            tool_choice_context_dict
        )
        return tool, tool_thoughts, tool_docstring

    responses = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_judge) for _ in range(3)]
        for future in as_completed(futures):
            responses.append(future.result())

    responses = [r for r in responses if r[0] is not None]
    counts: Dict[str, int] = {}
    for tool, tool_thoughts, tool_docstring in responses:
        if tool is not None:
            counts[tool.__name__] = counts.get(tool.__name__, 0) + 1
            if counts[tool.__name__] >= 2:
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
    if tool in TOOL_FUNCTIONS:
        tool = TOOL_FUNCTIONS[tool]
        tool_docstring = T.TOOLS_INFO[tool.__name__]

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


def run_tool_testing(
    task: str,
    image_paths: List[str],
    lmm: LMM,
    exclude_tools: Optional[List[str]],
    code_interpreter: CodeInterpreter,
    process_code: Callable[[str], str] = lambda x: x,
) -> tuple[str, str, Execution]:
    """Helper function to generate and run tool testing code."""
    query = lmm.generate(CATEGORIZE_TOOL_REQUEST.format(task=task))
    category = extract_tag(query, "category")  # type: ignore
    if category is None:
        query = task
    else:
        query = f"{category.strip()}. {task}"

    tool_docs = get_tool_recommender().top_k(query, k=5, thresh=0.3)
    if exclude_tools is not None and len(exclude_tools) > 0:
        cleaned_tool_docs = []
        for tool_doc in tool_docs:
            if not tool_doc["name"] in exclude_tools:
                cleaned_tool_docs.append(tool_doc)
        tool_docs = cleaned_tool_docs
    tool_docs_str = "\n".join([e["doc"] for e in tool_docs])
    tool_docs_str += "\n" + LOAD_TOOLS_DOCSTRING

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
        code = extract_tag(response, "code")
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
        task: str: The task to accomplish.
        images: Union[Dict[str, List[np.ndarray]], List[np.ndarray]]: The images to use
            for the task. If a key is provided, it is used as the file name.
        exclude_tools: Optional[List[str]]: A list of tool names to exclude from the
            recommendations. This is helpful if you are calling get_tool_for_task twice
            and do not want the same tool recommended.

    Returns:
        The tool to use for the task is printed to stdout

    Examples
    --------
        >>> get_tool_for_task(
        >>>     "Give me an OCR model that can find 'hot chocolate' in the image",
        >>>     {"image": [image]})
        >>> get_tool_for_taks(
        >>>     "I need a tool that can paint a background for this image and maks",
        >>>     {"image": [image], "mask": [mask]})
    """
    tool_tester = CONFIG.create_tool_tester()
    tool_chooser = CONFIG.create_tool_chooser()

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

        code, tool_docs_str, tool_output = run_tool_testing(
            task, image_paths, tool_tester, exclude_tools, code_interpreter
        )
        tool_output_str = tool_output.text(include_results=False).strip()

        _, tool_thoughts, tool_docstring = run_multi_judge(
            tool_chooser, tool_docs_str, task, code, tool_output_str, image_paths
        )

    print(format_tool_output(tool_thoughts, tool_docstring))


def get_tool_documentation(tool_name: str) -> str:
    # use same format as get_tool_for_task
    tool_doc = T.TOOLS_DF[T.TOOLS_DF["name"] == tool_name]["doc"].values[0]
    return format_tool_output("", tool_doc)


def get_tool_for_task_human_reviewer(
    task: str,
    images: Union[Dict[str, List[np.ndarray]], List[np.ndarray]],
    exclude_tools: Optional[List[str]] = None,
) -> None:
    # NOTE: this will have the same documentation as get_tool_for_task
    tool_tester = CONFIG.create_tool_tester()

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

        tools = [
            t.__name__
            for t in T.TOOLS
            if inspect.signature(t).parameters.get("box_threshold")  # type: ignore
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


def claude35_vqa(prompt: str, medias: List[np.ndarray]) -> None:
    """Asks the Claude-3.5 model a question about the given media and returns an answer.

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
]
PLANNER_DOCSTRING = T.get_tool_documentation(PLANNER_TOOLS)  # type: ignore
