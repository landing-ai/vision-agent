import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union, cast

import requests
from PIL.Image import Image as ImageType

from vision_agent.image_utils import convert_to_b64

_LOGGER = logging.getLogger(__name__)


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
    _ENDPOINT = "https://chnicr4kes5ku77niv2zoytggq0qyqlp.lambda-url.us-east-2.on.aws"

    doc = (
        "Grounding DINO is a tool that can detect arbitrary objects with inputs such as category names or referring expressions."
        "Here are some exmaples of how to use the tool, the examples are in the format of User Question: which will have the user's question in quotes followed by the parameters in JSON format, which is the parameters you need to output to call the API to solve the user's question.\n"
        'Example 1: User Question: "Can you build me a car detector?" {{"Parameters":{{"prompt": "car"}}}}\n'
        'Example 2: User Question: "Can you detect the person on the left?" {{"Parameters":{{"prompt": "person on the left"}}\n'
        'Exmaple 3: User Question: "Can you build me a tool that detects red shirts and green shirts?" {{"Parameters":{{"prompt": "red shirt. green shirt"}}}}\n'
    )

    def __init__(self, prompt: str):
        self.prompt = prompt

    def __call__(self, image: Union[str, Path, ImageType]) -> List[Dict]:
        image_b64 = convert_to_b64(image)
        data = {
            "prompt": self.prompt,
            "images": [image_b64],
        }
        res = requests.post(
            self._ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=data,
        )
        resp_json: Dict[str, Any] = res.json()
        if resp_json["statusCode"] != 200:
            _LOGGER.error(f"Request failed: {resp_json}")
        return cast(List[Dict], resp_json["data"])


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
