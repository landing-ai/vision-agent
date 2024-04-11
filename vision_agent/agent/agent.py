from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union


class Agent(ABC):
    @abstractmethod
    def __call__(
        self,
        input: Union[List[Dict[str, str]], str],
        image: Optional[Union[str, Path]] = None,
    ) -> str:
        pass

    @abstractmethod
    def log_progress(self, description: str) -> None:
        """Log the progress of the agent.
        This is a hook that is intended for reporting the progress of the agent.
        """
        pass
