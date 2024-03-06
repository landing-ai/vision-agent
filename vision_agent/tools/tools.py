from typing import Dict, List, Union
from abc import ABC, abstractmethod

from PIL.Image import Image as ImageType


class ImageTool(ABC):
    @abstractmethod
    def __call__(self, image: Union[str, ImageType]) -> List[Dict]:
        pass


class CLIP(ImageTool):
    doc = (
        "CLIP is a tool that can classify or tag any image given a set if input classes or tags."
        "Here are some exmaples of how to use the tool, the examples are in the format of User Question: which will have the user's question in quotes followed by the parameters in JSON format, which is the parameters you need to output to call the API to solve the user's question.\n"
        'Example 1: User Question: "Can you classify this image as a cat?" {{"Parameters":{{"prompt": ["cat"]}}}}\n'
        'Example 2: User Question: "Can you tag this photograph with cat or dog?" {{"Parameters":{{"prompt": ["cat", "dog"]}}}}\n'
        'Exmaple 3: User Question: "Can you build me a classifier taht classifies red shirts, green shirts and other?" {{"Parameters":{{"prompt": ["red shirt", "green shirt", "other"]}}}}\n'
    )

    def __init__(self, prompt: str):
        self.prompt = prompt

    def __call__(self, image: Union[str, ImageType]) -> List[Dict]:
        raise NotImplementedError


class GroundingDINO(ImageTool):
    doc = (
        "Grounding DINO is a tool that can detect arbitrary objects with inputs such as category names or referring expressions."
        "Here are some exmaples of how to use the tool, the examples are in the format of User Question: which will have the user's question in quotes followed by the parameters in JSON format, which is the parameters you need to output to call the API to solve the user's question.\n"
        'Example 1: User Question: "Can you build me a car detector?" {{"Parameters":{{"prompt": "car"}}}}\n'
        'Example 2: User Question: "Can you detect the person on the left?" {{"Parameters":{{"prompt": "person on the left"}}\n'
        'Exmaple 3: User Question: "Can you build me a tool that detects red shirts and green shirts?" {{"Parameters":{{"prompt": "red shirt. green shirt"}}}}\n'
    )

    def __init__(self, prompt: str):
        self.prompt = prompt

    def __call__(self, image: Union[str, ImageType]) -> List[Dict]:
        raise NotImplementedError


class GroundingSAM(ImageTool):
    doc = (
        "Grounding SAM is a tool that can detect and segment arbitrary objects with inputs such as category names or referring expressions."
        "Here are some exmaples of how to use the tool, the examples are in the format of User Question: which will have the user's question in quotes followed by the parameters in JSON format, which is the parameters you need to output to call the API to solve the user's question.\n"
        'Example 1: User Question: "Can you build me a car segmentor?" {{"Parameters":{{"prompt": "car"}}}}\n'
        'Example 2: User Question: "Can you segment the person on the left?" {{"Parameters":{{"prompt": "person on the left"}}\n'
        'Exmaple 3: User Question: "Can you build me a tool that segments red shirts and green shirts?" {{"Parameters":{{"prompt": "red shirt. green shirt"}}}}\n'
    )

    def __init__(self, prompt: str):
        self.prompt = prompt

    def __call__(self, image: Union[str, ImageType]) -> List[Dict]:
        raise NotImplementedError
