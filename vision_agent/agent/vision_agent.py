import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from vision_agent.agent import Agent
from vision_agent.agent.vision_agent_prompts import (
    EXAMPLES_CODE1,
    EXAMPLES_CODE2,
    EXAMPLES_CODE3,
    EXAMPLES_CODE3_EXTRA2,
    VA_CODE,
)
from vision_agent.lmm import LMM, AnthropicLMM, OpenAILMM
from vision_agent.models import Message
from vision_agent.tools.meta_tools import (
    META_TOOL_DOCSTRING,
    Artifacts,
    check_and_load_image,
    use_extra_vision_agent_args,
)
from vision_agent.utils import CodeInterpreterFactory
from vision_agent.utils.agent import extract_json, extract_tag
from vision_agent.utils.execute import CodeInterpreter, Execution

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)
WORKSPACE = Path(os.getenv("WORKSPACE", ""))
WORKSPACE.mkdir(parents=True, exist_ok=True)
if str(WORKSPACE) != "":
    os.environ["PYTHONPATH"] = f"{WORKSPACE}:{os.getenv('PYTHONPATH', '')}"


class BoilerplateCode:
    pre_code = [
        "from typing import *",
        "from vision_agent.utils.execute import CodeInterpreter",
        "from vision_agent.tools.meta_tools import Artifacts, open_code_artifact, create_code_artifact, edit_code_artifact, get_tool_descriptions, generate_vision_code, edit_vision_code, view_media_artifact, object_detection_fine_tuning, use_object_detection_fine_tuning, list_artifacts",
        "artifacts = Artifacts('{cwd}')",
    ]
    post_code: List[str] = []

    @staticmethod
    def add_boilerplate(code: str, **format: Any) -> str:
        """Run this method to prepend the default imports to the code.
        NOTE: be sure to run this method after the custom tools have been registered.
        """
        return (
            "\n".join([s.format(**format) for s in BoilerplateCode.pre_code])
            + "\n\n"
            + code
            + "\n\n"
            + "\n".join([s.format(**format) for s in BoilerplateCode.post_code])
        )


def format_agent_message(agent_message: str) -> str:
    agent_message_json = extract_json(agent_message)
    output = ""
    if "thinking" in agent_message_json and agent_message_json["thinking"]:
        output += "<thinking>" + agent_message_json["thinking"] + "</thinking>"
    if "response" in agent_message_json and agent_message_json["response"]:
        output += "<response>" + agent_message_json["response"] + "</response>"
    if "execute_python" in agent_message_json and agent_message_json["execute_python"]:
        output += (
            "\n<execute_python>\n"
            + agent_message_json["execute_python"]
            + "\n</execute_python>\n"
        )
    if (
        "let_user_respond" in agent_message_json
        and agent_message_json["let_user_respond"]
    ):
        output += (
            "<let_user_respond>"
            + str(agent_message_json["let_user_respond"])
            + "</let_user_respond>"
        )

    return output


def _clean_response(response: str) -> str:
    # Sometimes the LLM will hallucinate responses to an <execute_python> tag as if it
    # had already executed the code. This function removes the hallucinated response.
    if "<execute_python>" in response:
        end_execute_python = response.find("</execute_python>")
        response = response[: end_execute_python + len("</execute_python>")]
    return response


def run_conversation(orch: LMM, chat: List[Message]) -> Dict[str, Any]:
    chat = copy.deepcopy(chat)

    # only add 10 most recent messages in the chat to not go over token limit
    conversation = ""
    for chat_i in chat[-10:]:
        if chat_i["role"] == "user":
            conversation += f"USER: {chat_i['content']}\n\n"
        elif chat_i["role"] == "observation":
            conversation += f"OBSERVATION:\n{chat_i['content']}\n\n"
        elif chat_i["role"] == "assistant":
            conversation += f"AGENT: {format_agent_message(chat_i['content'])}\n\n"  # type: ignore
        else:
            raise ValueError(f"role {chat_i['role']} is not supported")

    prompt = VA_CODE.format(
        documentation=META_TOOL_DOCSTRING,
        examples=f"{EXAMPLES_CODE1}\n{EXAMPLES_CODE2}\n{EXAMPLES_CODE3}\n{EXAMPLES_CODE3_EXTRA2}",
        conversation=conversation,
    )
    message: Message = {"role": "user", "content": prompt}
    # only add recent media so we don't overload the model with old images
    if (
        chat[-1]["role"] == "observation"
        and "media" in chat[-1]
        and len(chat[-1]["media"]) > 0  # type: ignore
    ):
        media_obs = [media for media in chat[-1]["media"] if Path(media).exists()]  # type: ignore
        if len(media_obs) > 0:
            message["media"] = media_obs  # type: ignore
    conv_resp = cast(str, orch([message], stream=False))

    # clean the response first, if we are executing code, do not resond or end
    # conversation before the code has been executed.
    conv_resp = _clean_response(conv_resp)

    let_user_respond_str = extract_tag(conv_resp, "let_user_respond")
    let_user_respond = (
        "true" in let_user_respond_str.lower() if let_user_respond_str else False
    )

    return {
        "thinking": extract_tag(conv_resp, "thinking"),
        "response": extract_tag(conv_resp, "response"),
        "execute_python": extract_tag(conv_resp, "execute_python"),
        "let_user_respond": let_user_respond,
    }


def execute_code_action(
    artifacts: Artifacts,
    code: str,
    code_interpreter: CodeInterpreter,
) -> Tuple[Execution, str]:
    result = code_interpreter.exec_isolation(
        BoilerplateCode.add_boilerplate(code, cwd=str(artifacts.cwd))
    )

    obs = str(result.logs)
    if result.error:
        obs += f"\n{result.error}"
    return result, obs


def execute_user_code_action(
    artifacts: Artifacts,
    last_user_message: Message,
    code_interpreter: CodeInterpreter,
) -> Tuple[Optional[Execution], Optional[str]]:
    user_result = None
    user_obs = None

    if last_user_message["role"] != "user":
        return user_result, user_obs

    last_user_content = cast(str, last_user_message.get("content", ""))
    try:
        user_code_action = json.loads(last_user_content).get("execute_python", None)
    except json.JSONDecodeError:
        return user_result, user_obs

    if user_code_action is not None:
        user_code_action = use_extra_vision_agent_args(user_code_action, False)
        user_result, user_obs = execute_code_action(
            artifacts, user_code_action, code_interpreter
        )
        if user_result.error:
            user_obs += f"\n{user_result.error}"
    return user_result, user_obs


def add_step_descriptions(response: Dict[str, Any]) -> Dict[str, Any]:
    response = copy.deepcopy(response)

    if "execute_python" in response and response["execute_python"]:
        # only include descriptions for these, the rest will just have executing
        # code
        description_map = {
            "open_code_artifact": "Reading file.",
            "create_code_artifact": "Creating file.",
            "edit_code_artifact": "Editing file.",
            "generate_vision_code": "Generating vision code.",
            "edit_vision_code": "Editing vision code.",
        }
        description = ""
        for k, v in description_map.items():
            if k in response["execute_python"]:
                description += v + " "
        if description == "":
            description = "Executing code."

        response["response"] = description

    return response


def new_format_to_old_format(new_format: Dict[str, Any]) -> Dict[str, Any]:
    thoughts = new_format["thinking"] if new_format["thinking"] is not None else ""
    response = new_format["response"] if new_format["response"] is not None else ""
    if new_format["execute_python"] is not None:
        response += (
            f"\n<execute_python>\n{new_format['execute_python']}\n</execute_python>"
        )
    return {
        "thoughts": thoughts,
        "response": response,
        "let_user_respond": new_format["let_user_respond"],
    }


def old_format_to_new_format(old_format_str: str) -> str:
    try:
        old_format = json.loads(old_format_str)
    except json.JSONDecodeError:
        return old_format_str

    if "thoughts" in old_format:
        thinking = (
            old_format["thoughts"] if old_format["thoughts"].strip() != "" else None
        )
    else:
        thinking = None

    let_user_respond = (
        old_format["let_user_respond"] if "let_user_respond" in old_format else True
    )

    if "response" in old_format and "<execute_python>" in old_format["response"]:
        execute_python = extract_tag(old_format["response"], "execute_python")
        response = (
            old_format["response"]
            .replace(execute_python, "")
            .replace("<execute_python>", "")
            .replace("</execute_python>", "")
            .strip()
        )
    else:
        execute_python = None
        response = old_format["response"] if "response" in old_format else None

    return json.dumps(
        {
            "thinking": thinking,
            "response": response,
            "execute_python": execute_python,
            "let_user_respond": let_user_respond,
        }
    )


class VisionAgent(Agent):
    """Vision Agent is an agent that can chat with the user and call tools or other
    agents to generate code for it. Vision Agent uses python code to execute actions
    for the user. Vision Agent is inspired by by OpenDevin
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
        cwd: Optional[Union[Path, str]] = None,
        verbosity: int = 0,
        callback_message: Optional[Callable[[Dict[str, Any]], None]] = None,
        code_sandbox_runtime: Optional[str] = None,
    ) -> None:
        """Initialize the VisionAgent.

        Parameters:
            agent (Optional[LMM]): The agent to use for conversation and orchestration
                of other agents.
            verbosity (int): The verbosity level of the agent.
            callback_message (Optional[Callable[[Dict[str, Any]], None]]): Callback
                function to send intermediate update messages.
            code_sandbox_runtime (Optional[str]): For string values it can be one of:
                None or "local". If None, it will read from the environment
                variable "CODE_SANDBOX_RUNTIME".
        """

        self.agent = AnthropicLMM(temperature=0.0) if agent is None else agent
        self.max_iterations = 12
        self.cwd = Path(cwd) if cwd is not None else Path.cwd()
        self.verbosity = verbosity
        self.callback_message = callback_message
        self.code_sandbox_runtime = code_sandbox_runtime
        if self.verbosity >= 1:
            _LOGGER.setLevel(logging.INFO)

    def __call__(
        self,
        input: Union[str, List[Message]],
        media: Optional[Union[str, Path]] = None,
        artifacts: Optional[Artifacts] = None,
    ) -> str:
        """Chat with VisionAgent and get the conversation response.

        Parameters:
            input (Union[str, List[Message]): A conversation in the format of
                [{"role": "user", "content": "describe your task here..."}, ...] or a
                string of just the contents.
            media (Optional[Union[str, Path]]): The media file to be used in the task.
            artifacts (Optional[Artifacts]): The artifacts to use in the task.

        Returns:
            str: The conversation response.
        """
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
            if media is not None:
                input[0]["media"] = [media]
        results, _ = self.chat_with_artifacts(input, artifacts)
        return results[-1]["content"]  # type: ignore

    def chat(
        self,
        chat: List[Message],
    ) -> List[Message]:
        """Chat with VisionAgent, it will use code to execute actions to accomplish
        its tasks.

        Parameters:
            chat (List[Message]): A conversation in the format of:
                [{"role": "user", "content": "describe your task here..."}]
                or if it contains media files, it should be in the format of:
                [{"role": "user", "content": "describe your task here...", "media": ["image1.jpg", "image2.jpg"]}]

        Returns:
            List[Message]: The conversation response.
        """
        return self.chat_with_artifacts(chat)[0]

    def chat_with_artifacts(
        self,
        chat: List[Message],
        artifacts: Optional[Artifacts] = None,
        test_multi_plan: bool = True,
        custom_tool_names: Optional[List[str]] = None,
    ) -> Tuple[List[Message], Artifacts]:
        """Chat with VisionAgent, it will use code to execute actions to accomplish
        its tasks.

        Parameters:
            chat (List[Message]): A conversation in the format of:
                [{"role": "user", "content": "describe your task here..."}]
                or if it contains media files, it should be in the format of:
                [{"role": "user", "content": "describe your task here...", "media": ["image1.jpg", "image2.jpg"]}]
            artifacts (Optional[Artifacts]): The artifacts to use in the task.
            test_multi_plan (bool): If True, it will test tools for multiple plans and
                pick the best one based off of the tool results. If False, it will go
                with the first plan.
            custom_tool_names (List[str]): A list of customized tools for agent to
                pick and use. If not provided, default to full tool set from
                vision_agent.tools.

        Returns:
            List[Message]: The conversation response.
        """

        if not chat:
            raise ValueError("chat cannot be empty")

        if not artifacts:
            artifacts = Artifacts(self.cwd)

        with CodeInterpreterFactory.new_instance(
            code_sandbox_runtime=self.code_sandbox_runtime,
            remote_path=self.cwd,
        ) as code_interpreter:
            orig_chat = copy.deepcopy(chat)
            int_chat = copy.deepcopy(chat)
            last_user_message = chat[-1]
            for chat_i in int_chat:
                if "media" in chat_i:
                    for media in chat_i["media"]:
                        media = cast(str, media)
                        media_remote_path = Path(artifacts.cwd) / Path(media).name
                        chat_i["content"] += f" Media name {media_remote_path}"  # type: ignore

            int_chat = cast(
                List[Message],
                [
                    (
                        {
                            "role": c["role"],
                            "content": old_format_to_new_format(c["content"]),  # type: ignore
                            "media": c["media"],
                        }
                        if "media" in c
                        else {"role": c["role"], "content": old_format_to_new_format(c["content"])}  # type: ignore
                    )
                    for c in int_chat
                ],
            )

            finished = False
            iterations = 0
            last_response = None

            # Upload artifacts to remote location and show where they are going
            # to be loaded to. The actual loading happens in BoilerplateCode as
            # part of the pre_code.
            artifacts_loaded = artifacts.show()
            int_chat.append({"role": "observation", "content": artifacts_loaded})
            orig_chat.append({"role": "observation", "content": artifacts_loaded})
            self.streaming_message({"role": "observation", "content": artifacts_loaded})

            user_result, user_obs = execute_user_code_action(
                artifacts,
                last_user_message,
                code_interpreter,
            )
            finished = user_result is not None and user_obs is not None
            if user_result is not None and user_obs is not None:
                # be sure to update the chat with user execution results
                chat_elt: Message = {"role": "observation", "content": user_obs}
                int_chat.append(chat_elt)
                chat_elt["execution"] = user_result
                orig_chat.append(chat_elt)
                self.streaming_message(
                    {
                        "role": "observation",
                        "content": user_obs,
                        "execution": user_result,
                        "finished": finished,
                    }
                )

            while not finished and iterations < self.max_iterations:
                response = run_conversation(self.agent, int_chat)
                if self.verbosity >= 1:
                    _LOGGER.info(response)

                code_action = response.get("execute_python", None)
                # sometimes it gets stuck in a loop, so we force it to exit
                if last_response == response:
                    response["let_user_respond"] = True
                    self.streaming_message(
                        {
                            "role": "assistant",
                            "content": "{}",
                            "error": {
                                "name": "Error when running conversation agent",
                                "value": "Agent is stuck in conversation loop, exited",
                                "traceback_raw": [],
                            },
                            "finished": True,
                        }
                    )
                else:
                    self.streaming_message(
                        {
                            "role": "assistant",
                            "content": new_format_to_old_format(
                                add_step_descriptions(response)
                            ),
                            "finished": response.get("let_user_respond", False)
                            and code_action is None,
                        }
                    )

                int_chat.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            new_format_to_old_format(add_step_descriptions(response))
                        ),
                    }
                )
                orig_chat.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            new_format_to_old_format(add_step_descriptions(response))
                        ),
                    }
                )
                finished = response.get("let_user_respond", False)

                if code_action is not None:
                    code_action = use_extra_vision_agent_args(
                        code_action, test_multi_plan, custom_tool_names
                    )

                if code_action is not None:
                    result, obs = execute_code_action(
                        artifacts,
                        code_action,
                        code_interpreter,
                    )
                    obs_chat_elt: Message = {"role": "observation", "content": obs}
                    media_obs = check_and_load_image(code_action)
                    if media_obs and result.success:
                        obs_chat_elt["media"] = [
                            artifacts.cwd / media_ob for media_ob in media_obs
                        ]

                    if self.verbosity >= 1:
                        _LOGGER.info(obs)

                    # don't add execution results to internal chat
                    int_chat.append(obs_chat_elt)
                    obs_chat_elt["execution"] = result
                    orig_chat.append(obs_chat_elt)
                    self.streaming_message(
                        {
                            "role": "observation",
                            "content": obs,
                            "execution": result,
                            "finished": finished,
                        }
                    )

                iterations += 1
                last_response = response

        return orig_chat, artifacts

    def streaming_message(self, message: Dict[str, Any]) -> None:
        if self.callback_message:
            self.callback_message(message)

    def log_progress(self, data: Dict[str, Any]) -> None:
        pass


class OpenAIVisionAgent(VisionAgent):
    def __init__(
        self,
        agent: Optional[LMM] = None,
        cwd: Optional[Union[Path, str]] = None,
        verbosity: int = 0,
        callback_message: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Initialize the VisionAgent using OpenAI LMMs.

        Parameters:
            agent (Optional[LMM]): The agent to use for conversation and orchestration
                of other agents.
            verbosity (int): The verbosity level of the agent.
            callback_message (Optional[Callable[[Dict[str, Any]], None]]): Callback
                function to send intermediate update messages.
            code_interpreter (Optional[Union[str, CodeInterpreter]]): For string values
                it can be one of: None or "local". If None, it will read from
                the environment variable "CODE_SANDBOX_RUNTIME". If a CodeInterpreter
                object is provided it will use that.
        """

        agent = OpenAILMM(temperature=0.0, json_mode=True) if agent is None else agent
        super().__init__(
            agent,
            cwd,
            verbosity,
            callback_message,
        )


class AnthropicVisionAgent(VisionAgent):
    def __init__(
        self,
        agent: Optional[LMM] = None,
        cwd: Optional[Union[Path, str]] = None,
        verbosity: int = 0,
        callback_message: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Initialize the VisionAgent using Anthropic LMMs.

        Parameters:
            agent (Optional[LMM]): The agent to use for conversation and orchestration
                of other agents.
            verbosity (int): The verbosity level of the agent.
            callback_message (Optional[Callable[[Dict[str, Any]], None]]): Callback
                function to send intermediate update messages.
            code_interpreter (Optional[Union[str, CodeInterpreter]]): For string values
                it can be one of: None or "local". If None, it will read from
                the environment variable "CODE_SANDBOX_RUNTIME". If a CodeInterpreter
                object is provided it will use that.
        """

        agent = AnthropicLMM(temperature=0.0) if agent is None else agent
        super().__init__(
            agent,
            cwd,
            verbosity,
            callback_message,
        )
