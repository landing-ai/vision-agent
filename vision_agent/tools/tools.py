import logging
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, cast

import numpy as np
import requests
from PIL.Image import Image as ImageType

from vision_agent.image_utils import convert_to_b64, get_image_size

_LOGGER = logging.getLogger(__name__)


def normalize_bbox(
    bbox: List[Union[int, float]], image_size: Tuple[int, ...]
) -> List[float]:
    r"""Normalize the bounding box coordinates to be between 0 and 1."""
    x1, y1, x2, y2 = bbox
    x1 = round(x1 / image_size[1], 2)
    y1 = round(y1 / image_size[0], 2)
    x2 = round(x2 / image_size[1], 2)
    y2 = round(y2 / image_size[0], 2)
    return [x1, y1, x2, y2]


def rle_decode(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
    r"""Decode a run-length encoded mask. Returns numpy array, 1 - mask, 0 - background.

    Args:
        mask_rle: Run-length as string formated (start length)
        shape: The (height, width) of array to return
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


class Tool(ABC):
    name: str
    description: str
    usage: Dict


class CLIP(Tool):
    r"""CLIP is a tool that can classify or tag any image given a set if input classes
    or tags.

    Examples::
        >>> from vision_agent.tools import tools
        >>> t = tools.CLIP(["red line", "yellow dot", "none"])
        >>> t("examples/img/ct_scan1.jpg"))
        >>> [[0.02567436918616295, 0.9534115791320801, 0.020914122462272644]]
    """

    _ENDPOINT = "https://rb4ii6dfacmwqfxivi4aedyyfm0endsv.lambda-url.us-east-2.on.aws"

    name = "clip_"
    description = (
        "'clip_' is a tool that can classify or tag any image given a set if input classes or tags."
        "Here are some exmaples of how to use the tool, the examples are in the format of User Question: which will have the user's question in quotes followed by the parameters in JSON format, which is the parameters you need to output to call the API to solve the user's question.\n"
    )
    usage = {
        "required_parameters": [
            {"name": "prompt", "type": "List[str]"},
            {"name": "image", "type": "str"},
        ],
        "examples": [
            {
                "scenario": "Can you classify this image as a cat? Image name: cat.jpg",
                "parameters": {"prompt": ["cat"], "image": "cat.jpg"},
            },
            {
                "scenario": "Can you tag this photograph with cat or dog? Image name: cat_dog.jpg",
                "parameters": {"prompt": ["cat", "dog"], "image": "cat_dog.jpg"},
            },
            {
                "scenario": "Can you build me a classifier that classifies red shirts, green shirts and other? Image name: shirts.jpg",
                "parameters": {
                    "prompt": ["red shirt", "green shirt", "other"],
                    "image": "shirts.jpg",
                },
            },
        ],
    }

    def __call__(self, prompt: List[str], image: Union[str, ImageType]) -> List[Dict]:
        image_b64 = convert_to_b64(image)
        data = {
            "classes": prompt,
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
        return cast(List[Dict], resp_json["data"])


class GroundingDINO(Tool):
    _ENDPOINT = "https://chnicr4kes5ku77niv2zoytggq0qyqlp.lambda-url.us-east-2.on.aws"

    name = "grounding_dino_"
    description = (
        "'grounding_dino_' is a tool that can detect arbitrary objects with inputs such as category names or referring expressions."
        "Here are some exmaples of how to use the tool, the examples are in the format of User Question: which will have the user's question in quotes followed by the parameters in JSON format, which is the parameters you need to output to call the API to solve the user's question.\n"
        "The tool returns a list of dictionaries, each containing the following keys:\n"
        '  - "label": The label of the detected object.\n'
        '  - "score": The confidence score of the detection.\n'
        '  - "bbox": The bounding box of the detected object. The box coordinates are normalize to [0, 1]\n'
        'An example output would be: [{"label": ["car"], "score": [0.99], "bbox": [[0.1, 0.2, 0.3, 0.4]]}]\n'
    )
    usage = {
        "required_parameters": [
            {"name": "prompt", "type": "str"},
            {"name": "image", "type": "str"},
        ],
        "examples": [
            {
                "scenario": "Can you build me a car detector?",
                "parameters": {"prompt": "car", "image": ""},
            },
            {
                "scenario": "Can you detect the person on the left? Image name: person.jpg",
                "parameters": {"prompt": "person on the left", "image": "person.jpg"},
            },
            {
                "scenario": "Detect the red shirts and green shirst. Image name: shirts.jpg",
                "parameters": {
                    "prompt": "red shirt. green shirt",
                    "image": "shirts.jpg",
                },
            },
        ],
    }

    def __call__(self, prompt: str, image: Union[str, Path, ImageType]) -> List[Dict]:
        image_size = get_image_size(image)
        image_b64 = convert_to_b64(image)
        data = {
            "prompt": prompt,
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
            if "scores" in elt:
                elt["scores"] = [round(score, 2) for score in elt["scores"]]
        return cast(List[Dict], resp_data)


class GroundingSAM(Tool):
    r"""Grounding SAM is a tool that can detect and segment arbitrary objects with
    inputs such as category names or referring expressions.

    Examples::
        >>> from vision_agent.tools import tools
        >>> t = tools.GroundingSAM(["red line", "yellow dot", "none"])
        >>> t("examples/img/ct_scan1.jpg")
        >>> [{'label': 'none', 'mask': array([[0, 0, 0, ..., 0, 0, 0],
        >>>    [0, 0, 0, ..., 0, 0, 0],
        >>>    ...,
        >>>    [0, 0, 0, ..., 0, 0, 0],
        >>>    [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)}, {'label': 'red line', 'mask': array([[0, 0, 0, ..., 0, 0, 0],
        >>>    [0, 0, 0, ..., 0, 0, 0],
        >>>    ...,
        >>>    [1, 1, 1, ..., 1, 1, 1],
        >>>    [1, 1, 1, ..., 1, 1, 1]], dtype=uint8)}]
    """

    _ENDPOINT = "https://cou5lfmus33jbddl6hoqdfbw7e0qidrw.lambda-url.us-east-2.on.aws"

    name = "grounding_sam_"
    description = (
        "'grounding_sam_' is a tool that can detect and segment arbitrary objects with inputs such as category names or referring expressions."
        "Here are some exmaples of how to use the tool, the examples are in the format of User Question: which will have the user's question in quotes followed by the parameters in JSON format, which is the parameters you need to output to call the API to solve the user's question.\n"
    )
    usage = {
        "required_parameters": [
            {"name": "prompt", "type": "List[str]"},
            {"name": "image", "type": "str"},
        ],
        "examples": [
            {
                "scenario": "Can you build me a car segmentor?",
                "parameters": {"prompt": ["car"], "image": ""},
            },
            {
                "scenario": "Can you segment the person on the left? Image name: person.jpg",
                "parameters": {"prompt": ["person on the left"], "image": "person.jpg"},
            },
            {
                "scenario": "Can you build me a tool that segments red shirts and green shirts? Image name: shirts.jpg",
                "parameters": {
                    "prompt": ["red shirt", "green shirt"],
                    "image": "shirts.jpg",
                },
            },
        ],
    }

    def __call__(self, prompt: List[str], image: Union[str, ImageType]) -> List[Dict]:
        image_b64 = convert_to_b64(image)
        data = {
            "classes": prompt,
            "image": image_b64,
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
        preds = []
        for pred in resp_data["preds"]:
            encoded_mask = pred["encoded_mask"]
            mask = rle_decode(mask_rle=encoded_mask, shape=pred["mask_shape"])
            preds.append(
                {
                    "label": pred["label_name"],
                    "mask": mask,
                }
            )
        return preds


class Add(Tool):
    name = "add_"
    description = "'add_' returns the sum of all the arguments passed to it, normalized to 2 decimal places."
    usage = {
        "required_parameters": {"name": "input", "type": "List[int]"},
        "examples": [
            {
                "scenario": "If you want to calculate 2 + 4",
                "parameters": {"input": [2, 4]},
            }
        ],
    }

    def __call__(self, input: List[int]) -> float:
        return round(sum(input), 2)


class Subtract(Tool):
    name = "subtract_"
    description = "'subtract_' returns the difference of all the arguments passed to it, normalized to 2 decimal places."
    usage = {
        "required_parameters": {"name": "input", "type": "List[int]"},
        "examples": [
            {
                "scenario": "If you want to calculate 4 - 2",
                "parameters": {"input": [4, 2]},
            }
        ],
    }

    def __call__(self, input: List[int]) -> float:
        return round(input[0] - input[1], 2)


class Multiply(Tool):
    name = "multiply_"
    description = "'multiply_' returns the product of all the arguments passed to it, normalized to 2 decimal places."
    usage = {
        "required_parameters": {"name": "input", "type": "List[int]"},
        "examples": [
            {
                "scenario": "If you want to calculate 2 * 4",
                "parameters": {"input": [2, 4]},
            }
        ],
    }

    def __call__(self, input: List[int]) -> float:
        return round(input[0] * input[1], 2)


class Divide(Tool):
    name = "divide_"
    description = "'divide_' returns the division of all the arguments passed to it, normalized to 2 decimal places."
    usage = {
        "required_parameters": {"name": "input", "type": "List[int]"},
        "examples": [
            {
                "scenario": "If you want to calculate 4 / 2",
                "parameters": {"input": [4, 2]},
            }
        ],
    }

    def __call__(self, input: List[int]) -> float:
        return round(input[0] / input[1], 2)


TOOLS = {
    i: {"name": c.name, "description": c.description, "usage": c.usage, "class": c}
    for i, c in enumerate(
        [CLIP, GroundingDINO, GroundingSAM, Add, Subtract, Multiply, Divide]
    )
    if (hasattr(c, "name") and hasattr(c, "description") and hasattr(c, "usage"))
}
