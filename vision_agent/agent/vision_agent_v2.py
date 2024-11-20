import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

from vision_agent.agent import Agent, AgentCoder, VisionAgentCoderV2
from vision_agent.agent.agent_utils import add_media_to_chat, extract_tag
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
    if action == "plan_and_write_vision_code":
        extracted_chat = extract_conversation_for_generate_code(chat)
        # there's an issue here because coder.generate_code will send it's code_context
        # to the outside user via it's update_callback, but we don't necessarily have
        # access to that update_callback here, so we re-create the message using
        # format_code_context.
        code_context = coder.generate_code(extracted_chat, code_interpreter)
        return [
            AgentMessage(role="coder", content=format_code_context(code_context)),
            AgentMessage(role="observation", content=code_context.test_result.text()),
        ]
    elif action == "edit_vision_code":
        extracted_chat = extract_conversation_for_generate_code(chat)
        # place in a dummy plan because it just needs to edit the cord according to the
        # user's latest feedback.
        plan_context = PlanContext(
            plan="Edit the latest code observed according to the user's feedback.",
            instructions=[],
            code="",
        )
        code_context = coder.generate_code_from_plan(
            extracted_chat, plan_context, code_interpreter
        )
        return [
            AgentMessage(role="coder", content=format_code_context(code_context)),
            AgentMessage(role="observation", content=code_context.test_result.text()),
        ]
    elif action == "view_image":
        pass

    return None


class VisionAgentV2(Agent):
    def __init__(
        self,
        agent: Optional[LMM] = None,
        coder: Optional[AgentCoder] = None,
        verbose: bool = False,
        code_sandbox_runtime: Optional[str] = None,
        update_callback: Callable[[Dict[str, Any]], None] = lambda x: None,
    ) -> None:
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
        raise NotImplementedError

    def chat(
        self,
        chat: List[AgentMessage],
    ) -> List[AgentMessage]:
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
