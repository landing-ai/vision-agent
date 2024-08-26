import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from vision_agent.agent import Agent
from vision_agent.agent.agent_utils import extract_json
from vision_agent.agent.vision_agent_prompts import (
    EXAMPLES_CODE1,
    EXAMPLES_CODE2,
    VA_CODE,
)
from vision_agent.lmm import LMM, Message, OpenAILMM
from vision_agent.tools import META_TOOL_DOCSTRING
from vision_agent.utils import CodeInterpreterFactory
from vision_agent.utils.execute import CodeInterpreter

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)
WORKSPACE = Path(os.getenv("WORKSPACE", ""))
WORKSPACE.mkdir(parents=True, exist_ok=True)
if str(WORKSPACE) != "":
    os.environ["PYTHONPATH"] = f"{WORKSPACE}:{os.getenv('PYTHONPATH', '')}"


class DefaultImports:
    code = [
        "from typing import *",
        "from vision_agent.utils.execute import CodeInterpreter",
        "from vision_agent.tools.meta_tools import generate_vision_code, edit_vision_code, open_file, create_file, scroll_up, scroll_down, edit_file, get_tool_descriptions",
    ]

    @staticmethod
    def to_code_string() -> str:
        return "\n".join(DefaultImports.code)

    @staticmethod
    def prepend_imports(code: str) -> str:
        """Run this method to prepend the default imports to the code.
        NOTE: be sure to run this method after the custom tools have been registered.
        """
        return DefaultImports.to_code_string() + "\n\n" + code


def run_conversation(orch: LMM, chat: List[Message]) -> Dict[str, Any]:
    chat = copy.deepcopy(chat)

    conversation = ""
    for chat_i in chat:
        if chat_i["role"] == "user":
            conversation += f"USER: {chat_i['content']}\n\n"
        elif chat_i["role"] == "observation":
            conversation += f"OBSERVATION:\n{chat_i['content']}\n\n"
        elif chat_i["role"] == "assistant":
            conversation += f"AGENT: {chat_i['content']}\n\n"
        else:
            raise ValueError(f"role {chat_i['role']} is not supported")

    prompt = VA_CODE.format(
        documentation=META_TOOL_DOCSTRING,
        examples=f"{EXAMPLES_CODE1}\n{EXAMPLES_CODE2}",
        dir=WORKSPACE,
        conversation=conversation,
    )
    return extract_json(orch([{"role": "user", "content": prompt}], stream=False))  # type: ignore


def run_code_action(code: str, code_interpreter: CodeInterpreter) -> str:
    # Note the code interpreter needs to keep running in the same environment because
    # the SWE tools hold state like line numbers and currently open files.
    result = code_interpreter.exec_cell(DefaultImports.prepend_imports(code))

    return_str = ""
    if result.success:
        for res in result.results:
            if res.text is not None:
                return_str += res.text.replace("\\n", "\n")
        if result.logs.stdout:
            return_str += "----- stdout -----\n"
            for log in result.logs.stdout:
                return_str += log.replace("\\n", "\n")
    else:
        # for log in result.logs.stderr:
        #     return_str += log.replace("\\n", "\n")
        if result.error:
            return_str += (
                "\n" + result.error.value + "\n".join(result.error.traceback_raw)
            )

    return return_str


def parse_execution(response: str) -> Optional[str]:
    code = None
    if "<execute_python>" in response:
        code = response[response.find("<execute_python>") + len("<execute_python>") :]
        code = code[: code.find("</execute_python>")]
    return code


class VisionAgent(Agent):
    """Vision Agent is an agent that can chat with the user and call tools or other
    agents to generate code for it. Vision Agent uses python code to execute actions for
    the user. Vision Agent is inspired by by OpenDev
    https://github.com/OpenDevin/OpenDevin and CodeAct https://arxiv.org/abs/2402.01030

    Example
    -------
        >>> from vision_agent.agent import VisionAgent
        >>> agent = VisionAgent()
        >>> resp = agent("Hello")
        >>> resp.append({"role": "user", "content": "Can you write a function that counts dogs?", "media": ["dog.jpg"]})
        >>> resp = agent(resp)
    """

    def __init__(
        self,
        agent: Optional[LMM] = None,
        verbosity: int = 0,
        code_sandbox_runtime: Optional[str] = None,
    ) -> None:
        self.agent = (
            OpenAILMM(temperature=0.0, json_mode=True) if agent is None else agent
        )
        self.max_iterations = 100
        self.verbosity = verbosity
        self.code_sandbox_runtime = code_sandbox_runtime
        if self.verbosity >= 1:
            _LOGGER.setLevel(logging.INFO)

    def __call__(
        self,
        input: Union[str, List[Message]],
        media: Optional[Union[str, Path]] = None,
    ) -> str:
        """Chat with VisionAgent and get the conversation response.

        Parameters:
            input (Union[str, List[Message]): A conversation in the format of
                [{"role": "user", "content": "describe your task here..."}, ...] or a
                string of just the contents.
            media (Optional[Union[str, Path]]): The media file to be used in the task.

        Returns:
            str: The conversation response.
        """
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
            if media is not None:
                input[0]["media"] = [media]
        results = self.chat_with_code(input)
        return results  # type: ignore

    def chat_with_code(
        self,
        chat: List[Message],
    ) -> List[Message]:
        """Chat with VisionAgent, it will use code to execute actions to accomplish
        its tasks.

        Parameters:
            chat (List[Message]): A conversation
                in the format of:
                [{"role": "user", "content": "describe your task here..."}]
                or if it contains media files, it should be in the format of:
                [{"role": "user", "content": "describe your task here...", "media": ["image1.jpg", "image2.jpg"]}]

        Returns:
            List[Message]: The conversation response.
        """

        if not chat:
            raise ValueError("chat cannot be empty")

        with CodeInterpreterFactory.new_instance(
            code_sandbox_runtime=self.code_sandbox_runtime
        ) as code_interpreter:
            orig_chat = copy.deepcopy(chat)
            int_chat = copy.deepcopy(chat)
            media_list = []
            for chat_i in int_chat:
                if "media" in chat_i:
                    for media in chat_i["media"]:
                        media = code_interpreter.upload_file(media)
                        chat_i["content"] += f" Media name {media}"  # type: ignore
                        media_list.append(media)

            int_chat = cast(
                List[Message],
                [
                    (
                        {
                            "role": c["role"],
                            "content": c["content"],
                            "media": c["media"],
                        }
                        if "media" in c
                        else {"role": c["role"], "content": c["content"]}
                    )
                    for c in int_chat
                ],
            )

            finished = False
            iterations = 0
            while not finished and iterations < self.max_iterations:
                response = run_conversation(self.agent, int_chat)
                if self.verbosity >= 1:
                    _LOGGER.info(response)
                int_chat.append({"role": "assistant", "content": str(response)})
                orig_chat.append({"role": "assistant", "content": str(response)})

                if response["let_user_respond"]:
                    break

                code_action = parse_execution(response["response"])

                if code_action is not None:
                    obs = run_code_action(code_action, code_interpreter)
                    if self.verbosity >= 1:
                        _LOGGER.info(obs)
                    int_chat.append({"role": "observation", "content": obs})
                    orig_chat.append({"role": "observation", "content": obs})

                iterations += 1
        return orig_chat

    def log_progress(self, data: Dict[str, Any]) -> None:
        pass
