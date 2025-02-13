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
    interaction_response: The user's response to an interaction message.
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
        Literal["final_observation"],  # the observation from the final code output
        Literal["error_observation"],  # the observation from the error message
        Literal["interaction"],
        Literal["interaction_response"],
        Literal["conversation"],
        Literal["planner"],
        Literal[
            "planner_update"
        ],  # an intermediate update from the planner to show partial information
        Literal["coder"],
    ]
    content: str
    media: Optional[List[Union[str, Path]]] = None


class PlanContext(BaseModel):
    """PlanContext is a data model that represents the context of a plan.

    plan: A description of the overall plan.
    instructions: A list of step-by-step instructions.
    code: Code snippets that were used during planning.
    """

    plan: str
    instructions: List[str]
    code: str


class CodeContext(BaseModel):
    """CodeContext is a data model that represents final code and test cases.

    code: The final code that was written.
    test: The test cases that were written.
    success: A boolean value indicating whether the code passed the test cases.
    test_result: The result of running the test cases.
    """

    code: str
    test: str
    success: bool
    test_result: Execution


class InteractionContext(BaseModel):
    """InteractionContext is a data model that represents the context of an interaction.

    chat: A list of messages exchanged between the user and the assistant.
    """

    chat: List[AgentMessage]


class ErrorContext(BaseModel):
    """ErrorContext is a data model that represents an error message. These errors can
    happen in the planning phase when a model does not output correctly formatted
    messages (often because it considers some response to be a safety issue).

    error: The error message.
    """

    error: str
