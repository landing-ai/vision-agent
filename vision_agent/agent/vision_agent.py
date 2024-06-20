import copy
import difflib
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast

from langsmith import traceable
from PIL import Image
from rich.console import Console
from rich.style import Style
from rich.syntax import Syntax
from tabulate import tabulate

import vision_agent.tools as T
from vision_agent.agent import Agent
from vision_agent.agent.vision_agent_prompts import (
    CODE,
    FIX_BUG,
    FULL_TASK,
    PLAN,
    REFLECT,
    SIMPLE_TEST,
    USER_REQ,
)
from vision_agent.lmm import LMM, AzureOpenAILMM, Message, OpenAILMM
from vision_agent.utils import CodeInterpreterFactory, Execution
from vision_agent.utils.execute import CodeInterpreter
from vision_agent.utils.image_utils import b64_to_pil
from vision_agent.utils.sim import AzureSim, Sim
from vision_agent.utils.video import play_video

logging.basicConfig(stream=sys.stdout)
_LOGGER = logging.getLogger(__name__)
_MAX_TABULATE_COL_WIDTH = 80
_CONSOLE = Console()


class DefaultImports:
    """Container for default imports used in the code execution."""

    common_imports = [
        "from typing import *",
        "from pillow_heif import register_heif_opener",
        "register_heif_opener()",
    ]

    @staticmethod
    def to_code_string() -> str:
        return "\n".join(DefaultImports.common_imports + T.__new_tools__)

    @staticmethod
    def prepend_imports(code: str) -> str:
        """Run this method to prepend the default imports to the code.
        NOTE: be sure to run this method after the custom tools have been registered.
        """
        return DefaultImports.to_code_string() + "\n\n" + code


def get_diff(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True)
        )
    )


def format_memory(memory: List[Dict[str, str]]) -> str:
    output_str = ""
    for i, m in enumerate(memory):
        output_str += f"### Feedback {i}:\n"
        output_str += f"Code {i}:\n```python\n{m['code']}```\n\n"
        output_str += f"Feedback {i}: {m['feedback']}\n\n"
        if "edits" in m:
            output_str += f"Edits {i}:\n{m['edits']}\n"
        output_str += "\n"

    return output_str


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
        input_json_str = json_str
        if "```json" in json_str:
            json_str = json_str[json_str.find("```json") + len("```json") :]
            json_str = json_str[: json_str.find("```")]
        elif "```" in json_str:
            json_str = json_str[json_str.find("```") + len("```") :]
            # get the last ``` not one from an intermediate string
            json_str = json_str[: json_str.find("}```")]
        try:
            json_dict = json.loads(json_str)
        except json.JSONDecodeError as e:
            error_msg = f"Could not extract JSON from the given str: {json_str}.\nFunction input:\n{input_json_str}"
            _LOGGER.exception(error_msg)
            raise ValueError(error_msg) from e
    return json_dict  # type: ignore


def extract_image(
    media: Optional[Sequence[Union[str, Path]]]
) -> Optional[Sequence[Union[str, Path]]]:
    if media is None:
        return None

    new_media = []
    for m in media:
        m = Path(m)
        extension = m.suffix
        if extension in [".jpg", ".jpeg", ".png", ".bmp"]:
            new_media.append(m)
        elif extension in [".mp4", ".mov"]:
            frames = T.extract_frames(m)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                if len(frames) > 0:
                    Image.fromarray(frames[0][0]).save(tmp.name)
                    new_media.append(Path(tmp.name))
    if len(new_media) == 0:
        return None
    return new_media


@traceable
def write_plan(
    chat: List[Message],
    tool_desc: str,
    working_memory: str,
    model: LMM,
) -> List[Dict[str, str]]:
    chat = copy.deepcopy(chat)
    if chat[-1]["role"] != "user":
        raise ValueError("Last chat message must be from the user.")

    user_request = chat[-1]["content"]
    context = USER_REQ.format(user_request=user_request)
    prompt = PLAN.format(context=context, tool_desc=tool_desc, feedback=working_memory)
    chat[-1]["content"] = prompt
    return extract_json(model.chat(chat))["plan"]  # type: ignore


@traceable
def write_code(
    coder: LMM,
    chat: List[Message],
    tool_info: str,
    feedback: str,
) -> str:
    chat = copy.deepcopy(chat)
    if chat[-1]["role"] != "user":
        raise ValueError("Last chat message must be from the user.")

    user_request = chat[-1]["content"]
    prompt = CODE.format(
        docstring=tool_info,
        question=user_request,
        feedback=feedback,
    )
    chat[-1]["content"] = prompt
    return extract_code(coder(chat))


@traceable
def write_test(
    tester: LMM,
    chat: List[Message],
    tool_utils: str,
    code: str,
    feedback: str,
    media: Optional[Sequence[Union[str, Path]]] = None,
) -> str:
    chat = copy.deepcopy(chat)
    if chat[-1]["role"] != "user":
        raise ValueError("Last chat message must be from the user.")

    user_request = chat[-1]["content"]
    prompt = SIMPLE_TEST.format(
        docstring=tool_utils,
        question=user_request,
        code=code,
        feedback=feedback,
        media=media,
    )
    chat[-1]["content"] = prompt
    return extract_code(tester(chat))


@traceable
def reflect(
    chat: List[Message],
    plan: str,
    code: str,
    model: LMM,
) -> Dict[str, Union[str, bool]]:
    chat = copy.deepcopy(chat)
    if chat[-1]["role"] != "user":
        raise ValueError("Last chat message must be from the user.")

    user_request = chat[-1]["content"]
    context = USER_REQ.format(user_request=user_request)
    prompt = REFLECT.format(context=context, plan=plan, code=code)
    chat[-1]["content"] = prompt
    return extract_json(model(chat))


def write_and_test_code(
    chat: List[Message],
    tool_info: str,
    tool_utils: str,
    working_memory: List[Dict[str, str]],
    coder: LMM,
    tester: LMM,
    debugger: LMM,
    code_interpreter: CodeInterpreter,
    log_progress: Callable[[Dict[str, Any]], None],
    verbosity: int = 0,
    max_retries: int = 3,
    media: Optional[Sequence[Union[str, Path]]] = None,
) -> Dict[str, Any]:
    log_progress(
        {
            "type": "code",
            "status": "started",
        }
    )
    code = write_code(coder, chat, tool_info, format_memory(working_memory))
    test = write_test(
        tester, chat, tool_utils, code, format_memory(working_memory), media
    )

    log_progress(
        {
            "type": "code",
            "status": "running",
            "payload": {
                "code": DefaultImports.prepend_imports(code),
                "test": test,
            },
        }
    )
    result = code_interpreter.exec_isolation(
        f"{DefaultImports.to_code_string()}\n{code}\n{test}"
    )
    log_progress(
        {
            "type": "code",
            "status": "completed" if result.success else "failed",
            "payload": {
                "code": DefaultImports.prepend_imports(code),
                "test": test,
                "result": result.to_json(),
            },
        }
    )
    if verbosity == 2:
        _print_code("Initial code and tests:", code, test)
        _LOGGER.info(
            f"Initial code execution result:\n{result.text(include_logs=True)}"
        )

    count = 0
    new_working_memory: List[Dict[str, str]] = []
    while not result.success and count < max_retries:
        if verbosity == 2:
            _LOGGER.info(f"Start debugging attempt {count + 1}")
        code, test, result = debug_code(
            working_memory,
            debugger,
            code_interpreter,
            code,
            test,
            result,
            new_working_memory,
            log_progress,
            verbosity,
        )
        count += 1

    if verbosity >= 1:
        _print_code("Final code and tests:", code, test)

    return {
        "code": code,
        "test": test,
        "success": result.success,
        "test_result": result,
        "working_memory": new_working_memory,
    }


@traceable
def debug_code(
    working_memory: List[Dict[str, str]],
    debugger: LMM,
    code_interpreter: CodeInterpreter,
    code: str,
    test: str,
    result: Execution,
    new_working_memory: List[Dict[str, str]],
    log_progress: Callable[[Dict[str, Any]], None],
    verbosity: int = 0,
) -> tuple[str, str, Execution]:
    log_progress(
        {
            "type": "code",
            "status": "started",
        }
    )

    fixed_code_and_test = {"code": "", "test": "", "reflections": ""}
    success = False
    count = 0
    while not success and count < 3:
        try:
            fixed_code_and_test = extract_json(
                debugger(
                    FIX_BUG.format(
                        code=code,
                        tests=test,
                        result="\n".join(result.text().splitlines()[-50:]),
                        feedback=format_memory(working_memory + new_working_memory),
                    )
                )
            )
            success = True
        except Exception as e:
            _LOGGER.exception(f"Error while extracting JSON: {e}")

        count += 1

    old_code = code
    old_test = test

    if fixed_code_and_test["code"].strip() != "":
        code = extract_code(fixed_code_and_test["code"])
    if fixed_code_and_test["test"].strip() != "":
        test = extract_code(fixed_code_and_test["test"])

    new_working_memory.append(
        {
            "code": f"{code}\n{test}",
            "feedback": fixed_code_and_test["reflections"],
            "edits": get_diff(f"{old_code}\n{old_test}", f"{code}\n{test}"),
        }
    )
    log_progress(
        {
            "type": "code",
            "status": "running",
            "payload": {
                "code": DefaultImports.prepend_imports(code),
                "test": test,
            },
        }
    )

    result = code_interpreter.exec_isolation(
        f"{DefaultImports.to_code_string()}\n{code}\n{test}"
    )
    log_progress(
        {
            "type": "code",
            "status": "completed" if result.success else "failed",
            "payload": {
                "code": DefaultImports.prepend_imports(code),
                "test": test,
                "result": result.to_json(),
            },
        }
    )
    if verbosity == 2:
        _print_code("Code and test after attempted fix:", code, test)
        _LOGGER.info(
            f"Reflection: {fixed_code_and_test['reflections']}\nCode execution result after attempted fix: {result.text(include_logs=True)}"
        )

    return code, test, result


def _print_code(title: str, code: str, test: Optional[str] = None) -> None:
    _CONSOLE.print(title, style=Style(bgcolor="dark_orange3", bold=True))
    _CONSOLE.print("=" * 30 + " Code " + "=" * 30)
    _CONSOLE.print(
        Syntax(
            DefaultImports.prepend_imports(code),
            "python",
            theme="gruvbox-dark",
            line_numbers=True,
        )
    )
    if test:
        _CONSOLE.print("=" * 30 + " Test " + "=" * 30)
        _CONSOLE.print(Syntax(test, "python", theme="gruvbox-dark", line_numbers=True))


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
    tool_list: List[Dict[str, str]] = []
    for task in plan:
        tools = tool_recommender.top_k(task["instructions"], k=2, thresh=0.3)
        tool_info.extend([e["doc"] for e in tools])
        tool_desc.extend([e["desc"] for e in tools])
        tool_list.extend(
            {"description": e["desc"], "documentation": e["doc"]} for e in tools
        )
    log_progress(
        {
            "type": "tools",
            "status": "completed",
            "payload": {v["description"]: v for v in tool_list}.values(),
        }
    )

    if verbosity == 2:
        tool_desc_str = "\n".join(set(tool_desc))
        _LOGGER.info(f"Tools Description:\n{tool_desc_str}")
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
        planner: Optional[LMM] = None,
        coder: Optional[LMM] = None,
        tester: Optional[LMM] = None,
        debugger: Optional[LMM] = None,
        tool_recommender: Optional[Sim] = None,
        verbosity: int = 0,
        report_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Initialize the Vision Agent.

        Parameters:
            planner (Optional[LMM]): The planner model to use. Defaults to OpenAILMM.
            coder (Optional[LMM]): The coder model to use. Defaults to OpenAILMM.
            tester (Optional[LMM]): The tester model to use. Defaults to OpenAILMM.
            debugger (Optional[LMM]): The debugger model to
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
            OpenAILMM(temperature=0.0, json_mode=True) if planner is None else planner
        )
        self.coder = OpenAILMM(temperature=0.0) if coder is None else coder
        self.tester = OpenAILMM(temperature=0.0) if tester is None else tester
        self.debugger = (
            OpenAILMM(temperature=0.0, json_mode=True) if debugger is None else debugger
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
        input: Union[str, List[Message]],
        media: Optional[Union[str, Path]] = None,
    ) -> str:
        """Chat with Vision Agent and return intermediate information regarding the task.

        Parameters:
            input (Union[List[Dict[str, str]], str]): A conversation in the format of
                [{"role": "user", "content": "describe your task here..."}] or a string
                of just the contents.
            media (Optional[Union[str, Path]]): The media file to be used in the task.

        Returns:
            str: The code output by the Vision Agent.
        """

        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
            if media is not None:
                input[0]["media"] = [media]
        results = self.chat_with_workflow(input)
        results.pop("working_memory")
        return results  # type: ignore

    @traceable
    def chat_with_workflow(
        self,
        chat: List[Message],
        self_reflection: bool = False,
        display_visualization: bool = False,
    ) -> Dict[str, Any]:
        """Chat with Vision Agent and return intermediate information regarding the task.

        Parameters:
            chat (List[MediaChatItem]): A conversation
                in the format of:
                [{"role": "user", "content": "describe your task here..."}]
                or if it contains media files, it should be in the format of:
                [{"role": "user", "content": "describe your task here...", "media": ["image1.jpg", "image2.jpg"]}]
            self_reflection (bool): Whether to reflect on the task and debug the code.
            display_visualization (bool): If True, it opens a new window locally to
                show the image(s) created by visualization code (if there is any).

        Returns:
            Dict[str, Any]: A dictionary containing the code, test, test result, plan,
                and working memory of the agent.
        """

        if not chat:
            raise ValueError("Chat cannot be empty.")

        # NOTE: each chat should have a dedicated code interpreter instance to avoid concurrency issues
        with CodeInterpreterFactory.new_instance() as code_interpreter:
            chat = copy.deepcopy(chat)
            media_list = []
            for chat_i in chat:
                if "media" in chat_i:
                    for media in chat_i["media"]:
                        media = code_interpreter.upload_file(media)
                        chat_i["content"] += f" Media name {media}"  # type: ignore
                        media_list.append(media)

            int_chat = cast(
                List[Message],
                [{"role": c["role"], "content": c["content"]} for c in chat],
            )

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
                    int_chat,
                    T.TOOL_DESCRIPTIONS,
                    format_memory(working_memory),
                    self.planner,
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
                        f"\n{tabulate(tabular_data=plan_i, headers='keys', tablefmt='mixed_grid', maxcolwidths=_MAX_TABULATE_COL_WIDTH)}"
                    )

                tool_info = retrieve_tools(
                    plan_i,
                    self.tool_recommender,
                    self.log_progress,
                    self.verbosity,
                )
                results = write_and_test_code(
                    chat=int_chat,
                    tool_info=tool_info,
                    tool_utils=T.UTILITIES_DOCSTRING,
                    working_memory=working_memory,
                    coder=self.coder,
                    tester=self.tester,
                    debugger=self.debugger,
                    code_interpreter=code_interpreter,
                    log_progress=self.log_progress,
                    verbosity=self.verbosity,
                    media=media_list,
                )
                success = cast(bool, results["success"])
                code = cast(str, results["code"])
                test = cast(str, results["test"])
                working_memory.extend(results["working_memory"])  # type: ignore
                plan.append({"code": code, "test": test, "plan": plan_i})

                if not self_reflection:
                    break

                self.log_progress(
                    {
                        "type": "self_reflection",
                        "status": "started",
                    }
                )
                reflection = reflect(
                    int_chat,
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

            execution_result = cast(Execution, results["test_result"])
            self.log_progress(
                {
                    "type": "final_code",
                    "status": "completed" if success else "failed",
                    "payload": {
                        "code": DefaultImports.prepend_imports(code),
                        "test": test,
                        "result": execution_result.to_json(),
                    },
                }
            )

            if display_visualization:
                for res in execution_result.results:
                    if res.png:
                        b64_to_pil(res.png).show()
                    if res.mp4:
                        play_video(res.mp4)

            return {
                "code": DefaultImports.prepend_imports(code),
                "test": test,
                "test_result": execution_result,
                "plan": plan,
                "working_memory": working_memory,
            }

    def log_progress(self, data: Dict[str, Any]) -> None:
        if self.report_progress_callback is not None:
            self.report_progress_callback(data)


class AzureVisionAgent(VisionAgent):
    """Vision Agent that uses Azure OpenAI APIs for planning, coding, testing.

    Pre-requisites:
    1. Set the environment variable AZURE_OPENAI_API_KEY to your Azure OpenAI API key.
    2. Set the environment variable AZURE_OPENAI_ENDPOINT to your Azure OpenAI endpoint.

    Example
    -------
        >>> from vision_agent import AzureVisionAgent
        >>> agent = AzureVisionAgent()
        >>> code = agent("What percentage of the area of the jar is filled with coffee beans?", media="jar.jpg")
    """

    def __init__(
        self,
        planner: Optional[LMM] = None,
        coder: Optional[LMM] = None,
        tester: Optional[LMM] = None,
        debugger: Optional[LMM] = None,
        tool_recommender: Optional[Sim] = None,
        verbosity: int = 0,
        report_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Initialize the Vision Agent.

        Parameters:
            planner (Optional[LMM]): The planner model to use. Defaults to OpenAILMM.
            coder (Optional[LMM]): The coder model to use. Defaults to OpenAILMM.
            tester (Optional[LMM]): The tester model to use. Defaults to OpenAILMM.
            debugger (Optional[LMM]): The debugger model to
            tool_recommender (Optional[Sim]): The tool recommender model to use.
            verbosity (int): The verbosity level of the agent. Defaults to 0. 2 is the
                highest verbosity level which will output all intermediate debugging
                code.
            report_progress_callback: a callback to report the progress of the agent.
                This is useful for streaming logs in a web application where multiple
                VisionAgent instances are running in parallel. This callback ensures
                that the progress are not mixed up.
        """
        super().__init__(
            planner=(
                AzureOpenAILMM(temperature=0.0, json_mode=True)
                if planner is None
                else planner
            ),
            coder=AzureOpenAILMM(temperature=0.0) if coder is None else coder,
            tester=AzureOpenAILMM(temperature=0.0) if tester is None else tester,
            debugger=(
                AzureOpenAILMM(temperature=0.0, json_mode=True)
                if debugger is None
                else debugger
            ),
            tool_recommender=(
                AzureSim(T.TOOLS_DF, sim_key="desc")
                if tool_recommender is None
                else tool_recommender
            ),
            verbosity=verbosity,
            report_progress_callback=report_progress_callback,
        )
