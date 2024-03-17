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
