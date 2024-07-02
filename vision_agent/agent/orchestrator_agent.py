import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import vision_agent as va
from vision_agent.agent import Agent
from vision_agent.agent.agent_utils import extract_json
from vision_agent.agent.orchestrator_agent_prompts import (
    EXAMPLES_CODE,
    EXAMPLES_JSON,
    ORCHESTRATOR_CODE,
    ORCHESTRATOR_JSON,
)
from vision_agent.lmm import LMM, Message, OpenAILMM
from vision_agent.tools import ORCH_TOOL_DOCSTRING
from vision_agent.utils import CodeInterpreterFactory, Execution
from vision_agent.utils.execute import CodeInterpreter

WORKSPACE = Path(os.getenv("WORKSPACE", ""))
WORKSPACE.mkdir(parents=True, exist_ok=True)
os.environ["PYTHONPATH"] = f"{os.getenv('PYTHONPATH', '')}:{WORKSPACE}"


class DefaultImports:
    code = [
        "from typing import *",
        "from vision_agent.utils.execute import CodeInterpreter",
        "from vision_agent.tools.orchestrator_tools import generate_vision_code, edit_vision_code",
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


def run_action_json(input_data: Dict[str, Union[str, Dict]]) -> Tuple[str, str]:
    if (
        "action" not in input_data
        or "argument" not in input_data
        or "thoughts" not in input_data
    ):
        raise ValueError("Input data malformatted, missing key")

    action: str = input_data["action"]
    argument: Dict[str, Any] = input_data["argument"]

    obs = None
    if action == "VisionAgent":
        agent = va.agent.VisionAgent()
        response = agent.chat_with_workflow(
            chat=[
                {
                    "role": "user",
                    "content": argument["prompt"],
                    "media": [argument["media"]],
                }
            ]
        )
        obs = f"<code>\n{response['code']}\n</code>\n<test>\n{response['test']}\n</test><output>\n{response['test_result'].results[0]['text/plain']}\n</output>"
    elif action == "Respond":
        obs = argument["prompt"]
    else:
        raise ValueError(f"action {action} is not supported")

    return action, obs


def run_orchestrator_json(orch: LMM, chat: List[Message]) -> Dict[str, Any]:
    chat = copy.deepcopy(chat)

    conversation = ""
    for chat_i in chat:
        if chat_i["role"] == "user":
            conversation += f"USER: {chat_i['content']}\n\n"
        elif chat_i["role"] == "observation":
            conversation += f"OBSERVATION:\n{chat_i['content']}\n\n"
        else:
            conversation += f"AGENT: {chat_i['content']}\n\n"

    prompt = ORCHESTRATOR_JSON.format(examples=EXAMPLES_JSON, conversation=conversation)
    response = extract_json(orch([{"role": "user", "content": prompt}]))
    return response


def run_orchestrator_code(orch: LMM, chat: List[Message]) -> Dict[str, Any]:
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

    prompt = ORCHESTRATOR_CODE.format(
        documentation=ORCH_TOOL_DOCSTRING,
        examples=EXAMPLES_CODE,
        conversation=conversation,
    )
    return extract_json(orch([{"role": "user", "content": prompt}]))


def run_code_action(code: str) -> str:
    with CodeInterpreterFactory.new_instance() as code_interpreter:
        result = code_interpreter.exec_isolation(DefaultImports.prepend_imports(code))

    return_str = ""
    for res in result.results:
        if res.text is not None:
            res_text = res.text.replace("\\n", "\n")
            if res_text.startswith("'") and res_text.endswith("'"):
                res_text = res_text[1:-1]
            return_str += res.text.replace("\\n", "\n")
    for log in result.logs.stdout:
        return_str += log.replace("\\n", "\n")
    for log in result.logs.stderr:
        return_str += log.replace("\\n", "\n")
    if result.error:
        return_str += "\n" + result.error.value
    return return_str


def parse_execution(response: str) -> Optional[str]:
    code = None
    if "<execute_python>" in response:
        code = response[response.find("<execute_python>") + len("<execute_python>") :]
        code = code[: code.find("</execute_python>")]
    return code


class OrchestratorAgent(Agent):
    def __init__(
        self,
        agent: Optional[LMM] = None,
    ) -> None:
        self.agent = (
            OpenAILMM(temperature=0.0, json_mode=True) if agent is None else agent
        )
        self.max_iterations = 100

    def __call__(
        self,
        input: Union[str, List[Message]],
        media: Optional[Union[str, Path]] = None,
    ):
        return "This is a planner agent"

    def chat_in_code(
        self,
        chat: List[Message],
    ) -> List[Message]:
        if not chat:
            raise ValueError("chat cannot be empty")

        orig_chat = copy.deepcopy(chat)
        int_chat = copy.deepcopy(chat)
        media_list = []
        for chat_i in int_chat:
            if "media" in chat_i:
                for media in chat_i["media"]:
                    chat_i["content"] += f" Media name {media}"  # type: ignore
                    media_list.append(media)

        int_chat = cast(
            List[Message],
            [{"role": c["role"], "content": c["content"]} for c in int_chat],
        )

        finished = False
        iterations = 0
        while not finished and iterations < self.max_iterations:
            response = run_orchestrator_code(self.agent, int_chat)
            print(response)
            int_chat.append({"role": "assistant", "content": str(response)})
            orig_chat.append({"role": "assistant", "content": str(response)})

            if response["let_user_respond"]:
                break

            code_action = parse_execution(response["response"])

            if code_action is not None:
                obs = run_code_action(code_action)
                print(obs)
                int_chat.append({"role": "observation", "content": obs})
                orig_chat.append({"role": "observation", "content": obs})

            iterations += 1
        return orig_chat

    def chat_in_json(
        self,
        chat: List[Message],
    ) -> List[Message]:
        if not chat:
            raise ValueError("chat cannot be empty")

        chat = copy.deepcopy(chat)
        media_list = []
        for chat_i in chat:
            if "media" in chat_i:
                for media in chat_i["media"]:
                    chat_i["content"] += f" Media name {media}"  # type: ignore
                    media_list.append(media)

        int_chat = cast(
            List[Message],
            [{"role": c["role"], "content": c["content"]} for c in chat],
        )

        finished = False
        while not finished:
            response = run_orchestrator_json(self.agent, int_chat)
            action, obs = run_action_json(response)
            if action == "VisionAgent":
                int_chat.append({"role": "observation", "content": obs})
                chat.append({"role": "assistant", "content": response["thoughts"]})
                chat.append({"role": "user", "content": obs})
            elif action == "Respond":
                int_chat.append({"role": "agent", "content": obs})
                chat.append({"role": "assistant", "content": obs})
                finished = True

        return chat

    def log_progress(self, data: Dict[str, Any]) -> None:
        pass
