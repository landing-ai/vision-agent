from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from vision_agent.lmm import MediaChatItem


class Agent(ABC):
    @abstractmethod
    def __call__(
        self,
        input: Union[str, List[MediaChatItem]],
    ) -> str:
        pass

    @abstractmethod
    def log_progress(self, data: Dict[str, Any]) -> None:
        """Log the progress of the agent.
        This is a hook that is intended for reporting the progress of the agent.
        """
        pass
