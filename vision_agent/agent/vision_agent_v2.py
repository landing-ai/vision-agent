import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from vision_agent.agent import Agent, AgentCoder, VisionAgentCoderV2
from vision_agent.agent.vision_agent_coder_v2 import format_code_context
from vision_agent.agent.vision_agent_prompts_v2 import CONVERSATION
from vision_agent.configs import Config
from vision_agent.lmm import LMM
from vision_agent.models import (
    AgentMessage,
    CodeContext,
    ErrorContext,
    InteractionContext,
    Message,
    PlanContext,
)
from vision_agent.utils.agent import (
    add_media_to_chat,
    convert_message_to_agentmessage,
    extract_tag,
    format_conversation,
)
from vision_agent.utils.execute import CodeInterpreter, CodeInterpreterFactory

CONFIG = Config()


def extract_conversation(
    chat: List[AgentMessage],
    include_conv: bool = False,
    include_errors: bool = False,
) -> Tuple[List[AgentMessage], Optional[str]]:
    chat = copy.deepcopy(chat)

    # if we are in the middle of an interaction, return all the intermediate planning
    # steps
    if check_for_interaction(chat):
        return chat, None

    extracted_chat = []
    for chat_i in chat:
        if chat_i.role == "user":
            extracted_chat.append(chat_i)
        elif chat_i.role == "coder":
            if "<final_code>" in chat_i.content:
                extracted_chat.append(chat_i)
        elif chat_i.role == "final_observation":
            extracted_chat.append(chat_i)
        elif include_conv and chat_i.role == "conversation":
            extracted_chat.append(chat_i)
        elif include_errors and chat_i.role == "error_observation":
            extracted_chat.append(chat_i)

    # only keep the last <final_code>, <final_test>
    final_code = None
    extracted_chat_strip_code: List[AgentMessage] = []
    for chat_i in reversed((extracted_chat)):
        # don't check role here because user could send updated <final_code>
        if "<final_code>" in chat_i.content and final_code is None:
            extracted_chat_strip_code = [chat_i] + extracted_chat_strip_code
            final_code = extract_tag(chat_i.content, "final_code")
            if final_code is not None:
                test_code = extract_tag(chat_i.content, "final_test")
                final_code += "\n" + test_code if test_code is not None else ""

        if "<final_code>" in chat_i.content and final_code is not None:
            continue

        extracted_chat_strip_code = [chat_i] + extracted_chat_strip_code

    return extracted_chat_strip_code, final_code


def run_conversation(agent: LMM, chat: List[AgentMessage]) -> str:
    # Include conversation and error messages. The error messages can come from one of
    # the agents refusing to write a correctly formatted message, want to inform the
    # conversation agent of this.
    extracted_chat, _ = extract_conversation(
        chat, include_conv=True, include_errors=True
    )

    conv = format_conversation(extracted_chat)
    prompt = CONVERSATION.format(
        conversation=conv,
    )
    response = agent([{"role": "user", "content": prompt}], stream=False)
    return cast(str, response)


def check_for_interaction(chat: List[AgentMessage]) -> bool:
    return (
        len(chat) > 2
        and chat[-2].role == "interaction"
        and chat[-1].role == "interaction_response"
    )


def maybe_run_action(
    coder: AgentCoder,
    action: Optional[str],
    chat: List[AgentMessage],
    code_interpreter: Optional[CodeInterpreter] = None,
) -> Optional[List[AgentMessage]]:
    extracted_chat, final_code = extract_conversation(chat)
    if action == "generate_or_edit_vision_code":
        # there's an issue here because coder.generate_code will send it's code_context
        # to the outside user via it's update_callback, but we don't necessarily have
        # access to that update_callback here, so we re-create the message using
        # format_code_context.
        context = coder.generate_code(extracted_chat, code_interpreter=code_interpreter)

        if isinstance(context, CodeContext):
            return [
                AgentMessage(role="coder", content=format_code_context(context)),
                AgentMessage(
                    role="final_observation", content=context.test_result.text()
                ),
            ]
        elif isinstance(context, InteractionContext):
            return [
                AgentMessage(
                    role="interaction",
                    content=json.dumps([elt.model_dump() for elt in context.chat]),
                )
            ]
        elif isinstance(context, ErrorContext):
            return [
                AgentMessage(role="error_observation", content=context.error),
            ]
    elif action == "edit_code":
        # We don't want to pass code in plan_context.code so the coder will generate
        # new code from plan_context.plan
        plan_context = PlanContext(
            plan="Edit the latest code observed in the fewest steps possible according to the user's feedback."
            + ("<code>\n" + final_code + "\n</code>" if final_code is not None else ""),
            instructions=[
                chat_i.content
                for chat_i in extracted_chat
                if chat_i.role == "user" and "<final_code>" not in chat_i.content
            ],
            code="",
        )

        context = coder.generate_code_from_plan(
            extracted_chat, plan_context, code_interpreter=code_interpreter
        )
        return [
            AgentMessage(role="coder", content=format_code_context(context)),
            AgentMessage(role="final_observation", content=context.test_result.text()),
        ]
    elif action == "view_image":
        pass

    return None


class VisionAgentV2(Agent):
    """VisionAgentV2 is a conversational agent that allows you to more easily use a
    coder agent such as VisionAgentCoderV2 to write vision code for you.
    """

    def __init__(
        self,
        agent: Optional[LMM] = None,
        coder: Optional[AgentCoder] = None,
        hil: bool = False,
        verbose: bool = False,
        code_sandbox_runtime: Optional[str] = None,
        update_callback: Callable[[Dict[str, Any]], None] = lambda x: None,
    ) -> None:
        """Initialize the VisionAgentV2.

        Parameters:
            agent (Optional[LMM]): The language model to use for the agent. If None, a
                default AnthropicLMM will be used.
            coder (Optional[AgentCoder]): The coder agent to use for generating vision
                code. If None, a default VisionAgentCoderV2 will be used.
            hil (bool): Whether to use human-in-the-loop mode.
            verbose (bool): Whether to print out debug information.
            code_sandbox_runtime (Optional[str]): The code sandbox runtime to use, can
                be one of: None, "local" or "e2b". If None, it will read from the
                environment variable CODE_SANDBOX_RUNTIME.
            update_callback (Callable[[Dict[str, Any]], None]): The callback function
                that will send back intermediate conversation messages.
        """

        self.agent = agent if agent is not None else CONFIG.create_agent()
        self.coder = (
            coder
            if coder is not None
            else VisionAgentCoderV2(
                verbose=verbose, update_callback=update_callback, hil=hil
            )
        )

        self.verbose = verbose
        self.code_sandbox_runtime = code_sandbox_runtime
        self.update_callback = update_callback

        # force coder to use the same update_callback
        if hasattr(self.coder, "update_callback"):
            self.coder.update_callback = update_callback

    def __call__(
        self,
        input: Union[str, List[Message]],
        media: Optional[Union[str, Path]] = None,
    ) -> str:
        """Conversational interface to the agent. This is the main method to use to
        interact with the agent. It takes in a string or list of messages and returns
        the agent's response as a string.

        Parameters:
            input (Union[str, List[Message]]): The input to the agent. This can be a
                string or a list of messages in the format of [{"role": "user",
                "content": "describe your task here..."}, ...].
            media (Optional[Union[str, Path]]): The path to the media file to use with
                the input. This can be an image or video file.

        Returns:
            str: The agent's response as a string.
        """

        input_msg = convert_message_to_agentmessage(input, media)
        return self.chat(input_msg)[-1].content

    def chat(
        self,
        chat: List[AgentMessage],
        code_interpreter: Optional[CodeInterpreter] = None,
    ) -> List[AgentMessage]:
        """Conversational interface to the agent. This is the main method to use to
        interact with the agent. It takes in a list of messages and returns the agent's
        response as a list of messages.

        Parameters:
            chat (List[AgentMessage]): The input to the agent. This should be a list of
                AgentMessage objects.
            code_interpreter (Optional[CodeInterpreter]): The code interpreter to use.

        Returns:
            List[AgentMessage]: The agent's response as a list of AgentMessage objects.
        """

        chat = copy.deepcopy(chat)
        if not chat or chat[-1].role not in {"user", "interaction_response"}:
            raise ValueError(
                f"Last chat message must be from the user or interaction_response, got {chat[-1].role}."
            )

        return_chat = []
        with (
            CodeInterpreterFactory.new_instance(self.code_sandbox_runtime)
            if code_interpreter is None
            else code_interpreter
        ) as code_interpreter:
            int_chat, _, _ = add_media_to_chat(chat, code_interpreter)

            # if we had an interaction and then recieved an observation from the user
            # go back into the same action to finish it.
            action = None
            if check_for_interaction(int_chat):
                action = "generate_or_edit_vision_code"
            else:
                response_context = run_conversation(self.agent, int_chat)
                return_chat.append(
                    AgentMessage(role="conversation", content=response_context)
                )
                self.update_callback(return_chat[-1].model_dump())
                action = extract_tag(response_context, "action")

            updated_chat = maybe_run_action(
                self.coder, action, int_chat, code_interpreter=code_interpreter
            )

            # return an interaction early to get users feedback
            if updated_chat is not None and updated_chat[-1].role == "interaction":
                return_chat.extend(updated_chat)
            elif updated_chat is not None and updated_chat[-1].role != "interaction":
                # do not append updated_chat to return_chat becuase the observation
                # from running the action will have already been added via the callbacks
                obs_response_context = run_conversation(
                    self.agent, int_chat + return_chat + updated_chat
                )
                return_chat.append(
                    AgentMessage(role="conversation", content=obs_response_context)
                )
                self.update_callback(return_chat[-1].model_dump())

        return return_chat

    def log_progress(self, data: Dict[str, Any]) -> None:
        pass
