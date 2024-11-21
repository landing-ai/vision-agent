from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from vision_agent.agent.types import AgentMessage, CodeContext, PlanContext
from vision_agent.lmm.types import Message
from vision_agent.utils.execute import CodeInterpreter


class Agent(ABC):
    @abstractmethod
    def __call__(
        self,
        input: Union[str, List[Message]],
        media: Optional[Union[str, Path]] = None,
    ) -> Union[str, List[Message]]:
        pass

    @abstractmethod
    def log_progress(self, data: Dict[str, Any]) -> None:
        """Log the progress of the agent.
        This is a hook that is intended for reporting the progress of the agent.
        """
        pass


class AgentCoder(Agent):
    @abstractmethod
    def generate_code(
        self,
        chat: List[AgentMessage],
        max_steps: Optional[int] = None,
        code_interpreter: Optional[CodeInterpreter] = None,
    ) -> CodeContext:
        pass

    @abstractmethod
    def generate_code_from_plan(
        self,
        chat: List[AgentMessage],
        plan_context: PlanContext,
        code_interpreter: Optional[CodeInterpreter] = None,
    ) -> CodeContext:
        pass


class AgentPlanner(Agent):
    @abstractmethod
    def generate_plan(
        self,
        chat: List[AgentMessage],
        max_steps: Optional[int] = None,
        code_interpreter: Optional[CodeInterpreter] = None,
    ) -> PlanContext:
        pass
