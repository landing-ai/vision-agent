import base64
import requests
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast
from lmm_tools.config import BASETEN_API_KEY, BASETEN_URL


def encode_image(image: Union[str, Path]) -> str:
    with open(image, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")
    return encoded_image


class LMM(ABC):
    @abstractmethod
    def generate(self, prompt: str, image: Optional[Union[str, Path]] = None) -> str:
        pass


class LLaVALMM(LMM):
    r"""An LMM class for the LLaVA-1.6 34B model."""

    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str, image: Optional[Union[str, Path]] = None) -> str:
        data = {"prompt": prompt}
        if image:
            data["image"] = encode_image(image)
        res = requests.post(
            BASETEN_URL,
            headers={"Authorization": f"Api-Key {BASETEN_API_KEY}"},
            json=data,
        )
        return res.text


class OpenAILMM(LMM):
    r"""An LMM class for the OpenAI GPT-4 Vision model."""

    def __init__(self, name: str):
        from openai import OpenAI

        self.name = name
        self.client = OpenAI()

    def generate(self, prompt: str, image: Optional[Union[str, Path]] = None) -> str:
        message: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if image:
            extension = Path(image).suffix
            encoded_image = encode_image(image)
            message[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{extension};base64,{encoded_image}",
                        "detail": "low",
                    },
                },
            )

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview", messages=message  # type: ignore
        )
        return cast(str, response.choices[0].message.content)


def get_lmm(name: str) -> LMM:
    if name == "openai":
        return OpenAILMM(name)
    elif name == "llava":
        return LLaVALMM(name)
    else:
        raise ValueError(f"Unknown LMM: {name}, current support openai, llava")
