from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import BaseModel

from vision_agent.utils.execute import Execution


class AgentMessage(BaseModel):
    """AgentMessage encompases messages sent to the entire Agentic system, which includes
    both LMMs and sub-agents.

    user: The user's message.
    assistant: The assistant's message.
    observation: An observation made after conducting an action, either by the user or
        assistant.
    interaction: An interaction between the user and the assistant. For example if the
        assistant wants to ask the user for help on a task, it could send an
        interaction message.
    conversation: Messages coming from the conversation agent, this is a type of
        assistant messages.
    planner: Messages coming from the planner agent, this is a type of assistant
        messages.
    coder: Messages coming from the coder agent, this is a type of assistant messages.

    """

    role: Union[
        Literal["user"],
        Literal["assistant"],  # planner, coder and conversation are of type assistant
        Literal["observation"],
        Literal["interaction"],
        Literal["conversation"],
        Literal["planner"],
        Literal["coder"],
    ]
    content: str
    media: Optional[List[Union[str, Path]]] = None


class PlanContext(BaseModel):
    plan: str
    instructions: List[str]
    code: str


class CodeContext(BaseModel):
    code: str
    test: str
    success: bool
    test_result: Execution
