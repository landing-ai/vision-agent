from abc import ABC, abstractmethod
from typing import Dict, List, Union


class Agent(ABC):
    @abstractmethod
    def __call__(self, input: Union[List[Dict[str, str]], str]) -> str:
        pass
