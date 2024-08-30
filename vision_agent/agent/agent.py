from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from vision_agent.lmm.types import Message


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
