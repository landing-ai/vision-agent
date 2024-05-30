from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class Agent(ABC):
    @abstractmethod
    def __call__(
        self,
        input: Union[List[Dict[str, str]], str],
        media: Optional[Union[str, Path]] = None,
    ) -> str:
        pass

    @abstractmethod
    def log_progress(self, data: Dict[str, Any]) -> None:
        """Log the progress of the agent.
        This is a hook that is intended for reporting the progress of the agent.
        """
        pass
