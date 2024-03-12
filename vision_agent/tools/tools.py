import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, cast

import requests
from PIL.Image import Image as ImageType

from vision_agent.image_utils import convert_to_b64, get_image_size

_LOGGER = logging.getLogger(__name__)


def normalize_bbox(
    bbox: List[Union[int, float]], image_size: Tuple[int, ...]
) -> List[float]:
    r"""Normalize the bounding box coordinates to be between 0 and 1."""
    x1, y1, x2, y2 = bbox
    x1 = x1 / image_size[1]
    y1 = y1 / image_size[0]
    x2 = x2 / image_size[1]
    y2 = y2 / image_size[0]
    return [x1, y1, x2, y2]


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
        "The tool returns a list of dictionaries, each containing the following keys:\n"
        "  - 'lable': The label of the detected object.\n"
        "  - 'score': The confidence score of the detection.\n"
        "  - 'bbox': The bounding box of the detected object. The box coordinates are normalize to [0, 1]\n"
        "An example output would be: [{'label': ['car'], 'score': [0.99], 'bbox': [[0.1, 0.2, 0.3, 0.4]]}]\n"
    )

    def __init__(self, prompt: str):
        self.prompt = prompt

    def __call__(self, image: Union[str, Path, ImageType]) -> List[Dict]:
        image_size = get_image_size(image)
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
        if (
            "statusCode" in resp_json and resp_json["statusCode"] != 200
        ) or "statusCode" not in resp_json:
            _LOGGER.error(f"Request failed: {resp_json}")
            raise ValueError(f"Request failed: {resp_json}")
        resp_data = resp_json["data"]
        for elt in resp_data:
            if "bboxes" in elt:
                elt["bboxes"] = [
                    normalize_bbox(box, image_size) for box in elt["bboxes"]
                ]
        return cast(List[Dict], resp_data)


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
