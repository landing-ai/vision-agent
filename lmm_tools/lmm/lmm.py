import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


def encode_image(image: str | Path) -> str:
    with open(image, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")
    return encoded_image


class LMM(ABC):
    @abstractmethod
    def generate(self, prompt: str, image: Optional[str | Path]) -> str:
        pass


class LLaVALMM(LMM):
    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str, image: Optional[str | Path]) -> str:
        pass


class OpenAILMM(LMM):
    def __init__(self, name: str):
        from openai import OpenAI

        self.name = name
        self.client = OpenAI()

    def generate(self, prompt: str, image: Optional[str | Path]) -> str:
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
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
            model="gpt-4-vision-preview",
            message=message
        )
        return response.choices[0].message.content


def get_lmm(name: str) -> LMM:
    if name == "openai":
        return OpenAILMM(name)
    elif name == "llava-v1.6-34b":
        return LLaVALMM(name)
    else:
        raise ValueError(f"Unknown LMM: {name}, current support openai, llava-v1.6-34b")
