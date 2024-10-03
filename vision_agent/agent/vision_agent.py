import copy
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from vision_agent.agent import Agent
from vision_agent.agent.agent_utils import extract_json
from vision_agent.agent.vision_agent_prompts import (
    EXAMPLES_CODE1,
    EXAMPLES_CODE2,
    EXAMPLES_CODE3,
    VA_CODE,
)
from vision_agent.lmm import LMM, AnthropicLMM, Message, OpenAILMM
from vision_agent.tools import META_TOOL_DOCSTRING
from vision_agent.tools.meta_tools import (
    Artifacts,
    check_and_load_image,
    use_extra_vision_agent_args,
)
from vision_agent.utils import CodeInterpreterFactory
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
        "from vision_agent.tools.meta_tools import Artifacts, open_code_artifact, create_code_artifact, edit_code_artifact, get_tool_descriptions, generate_vision_code, edit_vision_code, write_media_artifact, view_media_artifact, object_detection_fine_tuning, use_object_detection_fine_tuning",
        "artifacts = Artifacts('{remote_path}')",
        "artifacts.load('{remote_path}')",
    ]
    post_code = [
        "artifacts.save()",
    ]

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
        examples=f"{EXAMPLES_CODE1}\n{EXAMPLES_CODE2}\n{EXAMPLES_CODE3}",
        conversation=conversation,
    )
    message: Message = {"role": "user", "content": prompt}
    # only add recent media so we don't overload the model with old images
    if (
        chat[-1]["role"] == "observation"
        and "media" in chat[-1]
        and len(chat[-1]["media"]) > 0  # type: ignore
    ):
        message["media"] = chat[-1]["media"]
    return extract_json(orch([message], stream=False))  # type: ignore


def execute_code_action(
    code: str, code_interpreter: CodeInterpreter, artifact_remote_path: str
) -> Tuple[Execution, str]:
    result = code_interpreter.exec_isolation(
        BoilerplateCode.add_boilerplate(code, remote_path=artifact_remote_path)
    )

    obs = str(result.logs)
    if result.error:
        obs += f"\n{result.error}"
    return result, obs


def parse_execution(
    response: str,
    test_multi_plan: bool = True,
    customed_tool_names: Optional[List[str]] = None,
) -> Optional[str]:
    code = None
    remaining = response
    all_code = []
    while "<execute_python>" in remaining:
        code_i = remaining[
            remaining.find("<execute_python>") + len("<execute_python>") :
        ]
        code_i = code_i[: code_i.find("</execute_python>")]
        remaining = remaining[
            remaining.find("</execute_python>") + len("</execute_python>") :
        ]
        all_code.append(code_i)

    if len(all_code) > 0:
        code = "\n".join(all_code)

    if code is not None:
        code = use_extra_vision_agent_args(code, test_multi_plan, customed_tool_names)
    return code


def execute_user_code_action(
    last_user_message: Message,
    code_interpreter: CodeInterpreter,
    artifact_remote_path: str,
) -> Tuple[Optional[Execution], Optional[str]]:
    user_result = None
    user_obs = None

    if last_user_message["role"] != "user":
        return user_result, user_obs

    last_user_content = cast(str, last_user_message.get("content", ""))

    user_code_action = parse_execution(last_user_content, False)
    if user_code_action is not None:
        user_result, user_obs = execute_code_action(
            user_code_action, code_interpreter, artifact_remote_path
        )
        if user_result.error:
            user_obs += f"\n{user_result.error}"
    return user_result, user_obs


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
        verbosity: int = 0,
        local_artifacts_path: Optional[Union[str, Path]] = None,
        code_sandbox_runtime: Optional[str] = None,
        callback_message: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Initialize the VisionAgent.

        Parameters:
            agent (Optional[LMM]): The agent to use for conversation and orchestration
                of other agents.
            verbosity (int): The verbosity level of the agent.
            local_artifacts_path (Optional[Union[str, Path]]): The path to the local
                artifacts file.
            code_sandbox_runtime (Optional[str]): The code sandbox runtime to use.
        """

        self.agent = AnthropicLMM(temperature=0.0) if agent is None else agent
        self.max_iterations = 12
        self.verbosity = verbosity
        self.code_sandbox_runtime = code_sandbox_runtime
        self.callback_message = callback_message
        if self.verbosity >= 1:
            _LOGGER.setLevel(logging.INFO)
        self.local_artifacts_path = cast(
            str,
            (
                Path(local_artifacts_path)
                if local_artifacts_path is not None
                else Path(tempfile.NamedTemporaryFile(delete=False).name)
            ),
        )

    def __call__(
        self,
        input: Union[str, List[Message]],
        media: Optional[Union[str, Path]] = None,
        artifacts: Optional[Artifacts] = None,
    ) -> List[Message]:
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
        results, _ = self.chat_with_code(input, artifacts)
        return results

    def chat_with_code(
        self,
        chat: List[Message],
        artifacts: Optional[Artifacts] = None,
        test_multi_plan: bool = True,
        customized_tool_names: Optional[List[str]] = None,
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
            customized_tool_names (List[str]): A list of customized tools for agent to
                pick and use. If not provided, default to full tool set from
                vision_agent.tools.

        Returns:
            List[Message]: The conversation response.
        """

        if not chat:
            raise ValueError("chat cannot be empty")

        if not artifacts:
            # this is setting remote artifacts path
            artifacts = Artifacts(WORKSPACE / "artifacts.pkl")

        with CodeInterpreterFactory.new_instance(
            code_sandbox_runtime=self.code_sandbox_runtime,
        ) as code_interpreter:
            orig_chat = copy.deepcopy(chat)
            int_chat = copy.deepcopy(chat)
            last_user_message = chat[-1]
            media_list = []
            for chat_i in int_chat:
                if "media" in chat_i:
                    for media in chat_i["media"]:
                        media = cast(str, media)
                        artifacts.artifacts[Path(media).name] = open(media, "rb").read()

                        media_remote_path = (
                            Path(code_interpreter.remote_path) / Path(media).name
                        )
                        chat_i["content"] += f" Media name {media_remote_path}"  # type: ignore
                        media_list.append(media_remote_path)

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
            last_response = None

            # Save the current state of artifacts, will include any images the user
            # passed in.
            artifacts.save(self.local_artifacts_path)

            # Upload artifacts to remote location and show where they are going
            # to be loaded to. The actual loading happens in BoilerplateCode as
            # part of the pre_code.
            remote_artifacts_path = code_interpreter.upload_file(
                self.local_artifacts_path
            )
            artifacts_loaded = artifacts.show(code_interpreter.remote_path)
            int_chat.append({"role": "observation", "content": artifacts_loaded})
            orig_chat.append({"role": "observation", "content": artifacts_loaded})
            self.streaming_message({"role": "observation", "content": artifacts_loaded})

            user_result, user_obs = execute_user_code_action(
                last_user_message, code_interpreter, str(remote_artifacts_path)
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
                int_chat.append({"role": "assistant", "content": str(response)})
                orig_chat.append({"role": "assistant", "content": str(response)})

                # sometimes it gets stuck in a loop, so we force it to exit
                if last_response == response:
                    response["let_user_respond"] = True

                finished = response["let_user_respond"]

                code_action = parse_execution(
                    response["response"], test_multi_plan, customized_tool_names
                )

                if last_response == response:
                    self.streaming_message(
                        {
                            "role": "assistant",
                            "content": "{}",
                            "error": {
                                "name": "Error when running conversation agent",
                                "value": "Agent is stuck in conversation loop, exited",
                                "traceback_raw": [],
                            },
                            "finished": finished and code_action is None,
                        }
                    )
                else:
                    self.streaming_message(
                        {
                            "role": "assistant",
                            "content": response,
                            "finished": finished and code_action is None,
                        }
                    )

                if code_action is not None:
                    result, obs = execute_code_action(
                        code_action, code_interpreter, str(remote_artifacts_path)
                    )

                    media_obs = check_and_load_image(code_action)

                    if self.verbosity >= 1:
                        _LOGGER.info(obs)

                    obs_chat_elt: Message = {"role": "observation", "content": obs}
                    if media_obs and result.success:
                        obs_chat_elt["media"] = [
                            Path(code_interpreter.remote_path) / media_ob
                            for media_ob in media_obs
                        ]

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

            # after running the agent, download the artifacts locally
            code_interpreter.download_file(
                str(remote_artifacts_path.name), str(self.local_artifacts_path)
            )
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
        verbosity: int = 0,
        local_artifacts_path: Optional[Union[str, Path]] = None,
        code_sandbox_runtime: Optional[str] = None,
        callback_message: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Initialize the VisionAgent using OpenAI LMMs.

        Parameters:
            agent (Optional[LMM]): The agent to use for conversation and orchestration
                of other agents.
            verbosity (int): The verbosity level of the agent.
            local_artifacts_path (Optional[Union[str, Path]]): The path to the local
                artifacts file.
            code_sandbox_runtime (Optional[str]): The code sandbox runtime to use.
        """

        agent = OpenAILMM(temperature=0.0, json_mode=True) if agent is None else agent
        super().__init__(
            agent,
            verbosity,
            local_artifacts_path,
            code_sandbox_runtime,
            callback_message,
        )


class AnthropicVisionAgent(VisionAgent):
    def __init__(
        self,
        agent: Optional[LMM] = None,
        verbosity: int = 0,
        local_artifacts_path: Optional[Union[str, Path]] = None,
        code_sandbox_runtime: Optional[str] = None,
        callback_message: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Initialize the VisionAgent using Anthropic LMMs.

        Parameters:
            agent (Optional[LMM]): The agent to use for conversation and orchestration
                of other agents.
            verbosity (int): The verbosity level of the agent.
            local_artifacts_path (Optional[Union[str, Path]]): The path to the local
                artifacts file.
            code_sandbox_runtime (Optional[str]): The code sandbox runtime to use.
        """

        agent = AnthropicLMM(temperature=0.0) if agent is None else agent
        super().__init__(
            agent,
            verbosity,
            local_artifacts_path,
            code_sandbox_runtime,
            callback_message,
        )
