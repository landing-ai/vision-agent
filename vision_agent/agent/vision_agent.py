import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from vision_agent.agent import Agent
from vision_agent.agent.agent_utils import extract_json
from vision_agent.agent.vision_agent_prompts import EXAMPLES_CODE, VA_CODE
from vision_agent.lmm import LMM, Message, OpenAILMM
from vision_agent.tools import META_TOOL_DOCSTRING
from vision_agent.utils import CodeInterpreterFactory
from vision_agent.utils.execute import CodeInterpreter

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)
WORKSPACE = Path(os.getenv("WORKSPACE", ""))
WORKSPACE.mkdir(parents=True, exist_ok=True)
if WORKSPACE != "":
    os.environ["PYTHONPATH"] = f"{WORKSPACE}:{os.getenv('PYTHONPATH', '')}"


class DefaultImports:
    code = [
        "from typing import *",
        "from vision_agent.utils.execute import CodeInterpreter",
        "from vision_agent.tools.meta_tools import generate_vision_code, edit_vision_code, open_file, create_file, scroll_up, scroll_down, edit_file",
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
        examples=EXAMPLES_CODE,
        dir=WORKSPACE,
        conversation=conversation,
    )
    return extract_json(orch([{"role": "user", "content": prompt}]))


def run_code_action(code: str, code_interpreter: CodeInterpreter) -> str:
    result = code_interpreter.exec_isolation(DefaultImports.prepend_imports(code))

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
            return_str += "\n" + result.error.value

    return return_str


def parse_execution(response: str) -> Optional[str]:
    code = None
    if "<execute_python>" in response:
        code = response[response.find("<execute_python>") + len("<execute_python>") :]
        code = code[: code.find("</execute_python>")]
    return code

class VisionAgent(Agent):
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
    ) -> List[Message]:
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
                    for c in chat
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
                    _LOGGER.info(obs)
                    int_chat.append({"role": "observation", "content": obs})
                    orig_chat.append({"role": "observation", "content": obs})

                iterations += 1
        return orig_chat

    def log_progress(self, data: Dict[str, Any]) -> None:
        pass
