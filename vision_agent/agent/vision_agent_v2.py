import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

from vision_agent.agent import Agent, AgentCoder, VisionAgentCoderV2
from vision_agent.agent.agent_utils import (
    add_media_to_chat,
    convert_message_to_agentmessage,
    extract_tag,
)
from vision_agent.agent.types import AgentMessage, PlanContext
from vision_agent.agent.vision_agent_coder_v2 import format_code_context
from vision_agent.agent.vision_agent_prompts_v2 import CONVERSATION
from vision_agent.lmm import LMM, AnthropicLMM
from vision_agent.lmm.types import Message
from vision_agent.utils.execute import CodeInterpreter, CodeInterpreterFactory


def format_conversation(chat: List[AgentMessage]) -> str:
    chat = copy.deepcopy(chat)
    prompt = ""
    for chat_i in chat:
        if chat_i.role == "user":
            prompt += f"USER: {chat_i.content}\n\n"
        elif chat_i.role == "observation" or chat_i.role == "coder":
            prompt += f"OBSERVATION: {chat_i.content}\n\n"
        elif chat_i.role == "conversation":
            prompt += f"AGENT: {chat_i.content}\n\n"
    return prompt


def run_conversation(agent: LMM, chat: List[AgentMessage]) -> str:
    # only keep last 10 messages
    conv = format_conversation(chat[-10:])
    prompt = CONVERSATION.format(
        conversation=conv,
    )
    response = agent([{"role": "user", "content": prompt}], stream=False)
    return cast(str, response)


def extract_conversation_for_generate_code(
    chat: List[AgentMessage],
) -> List[AgentMessage]:
    chat = copy.deepcopy(chat)
    extracted_chat = []
    for chat_i in chat:
        if chat_i.role == "user":
            extracted_chat.append(chat_i)
        elif chat_i.role == "coder":
            if "<final_code>" in chat_i.content and "<final_test>" in chat_i.content:
                extracted_chat.append(chat_i)

    return extracted_chat


def maybe_run_action(
    coder: AgentCoder,
    action: Optional[str],
    chat: List[AgentMessage],
    code_interpreter: Optional[CodeInterpreter] = None,
) -> Optional[List[AgentMessage]]:
    if action == "generate_or_edit_vision_code":
        extracted_chat = extract_conversation_for_generate_code(chat)
        # there's an issue here because coder.generate_code will send it's code_context
        # to the outside user via it's update_callback, but we don't necessarily have
        # access to that update_callback here, so we re-create the message using
        # format_code_context.
        code_context = coder.generate_code(
            extracted_chat, code_interpreter=code_interpreter
        )
        return [
            AgentMessage(role="coder", content=format_code_context(code_context)),
            AgentMessage(role="observation", content=code_context.test_result.text()),
        ]
    elif action == "edit_code":
        extracted_chat = extract_conversation_for_generate_code(chat)
        plan_context = PlanContext(
            plan="Edit the latest code observed in the fewest steps possible according to the user's feedback.",
            instructions=[],
            code="",
        )
        code_context = coder.generate_code_from_plan(
            extracted_chat, plan_context, code_interpreter=code_interpreter
        )
        return [
            AgentMessage(role="coder", content=format_code_context(code_context)),
            AgentMessage(role="observation", content=code_context.test_result.text()),
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
            verbose (bool): Whether to print out debug information.
            code_sandbox_runtime (Optional[str]): The code sandbox runtime to use, can
                be one of: None, "local" or "e2b". If None, it will read from the
                environment variable CODE_SANDBOX_RUNTIME.
            update_callback (Callable[[Dict[str, Any]], None]): The callback function
                that will send back intermediate conversation messages.
        """

        self.agent = (
            agent
            if agent is not None
            else AnthropicLMM(
                model_name="claude-3-5-sonnet-20241022",
                temperature=0.0,
            )
        )
        self.coder = (
            coder
            if coder is not None
            else VisionAgentCoderV2(verbose=verbose, update_callback=update_callback)
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
    ) -> List[AgentMessage]:
        """Conversational interface to the agent. This is the main method to use to
        interact with the agent. It takes in a list of messages and returns the agent's
        response as a list of messages.

        Parameters:
            chat (List[AgentMessage]): The input to the agent. This should be a list of
                AgentMessage objects.

        Returns:
            List[AgentMessage]: The agent's response as a list of AgentMessage objects.
        """

        return_chat = []
        with CodeInterpreterFactory.new_instance(
            self.code_sandbox_runtime
        ) as code_interpreter:
            int_chat, _, _ = add_media_to_chat(chat, code_interpreter)
            response_context = run_conversation(self.agent, int_chat)
            return_chat.append(
                AgentMessage(role="conversation", content=response_context)
            )
            self.update_callback(return_chat[-1].model_dump())

            action = extract_tag(response_context, "action")

            updated_chat = maybe_run_action(
                self.coder, action, int_chat, code_interpreter=code_interpreter
            )
            if updated_chat is not None:
                # do not append updated_chat to return_chat becuase the observation
                # from running the action will have already been added via the callbacks
                obs_response_context = run_conversation(
                    self.agent, return_chat + updated_chat
                )
                return_chat.append(
                    AgentMessage(role="conversation", content=obs_response_context)
                )
                self.update_callback(return_chat[-1].model_dump())

        return return_chat

    def log_progress(self, data: Dict[str, Any]) -> None:
        pass
