import json
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Mapping, Union, cast

from openai import OpenAI

from vision_agent.tools import (
    CHOOSE_PARAMS,
    CLIP,
    SYSTEM_PROMPT,
    GroundingDINO,
    GroundingSAM,
)


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def chat(self, chat: List[Dict[str, str]]) -> str:
        pass

    @abstractmethod
    def __call__(self, input: Union[str, List[Dict[str, str]]]) -> str:
        pass


class OpenAILLM(LLM):
    r"""An LLM class for any OpenAI LLM model."""

    def __init__(
        self, model_name: str = "gpt-4-turbo-preview", json_mode: bool = False
    ):
        self.model_name = model_name
        self.client = OpenAI()
        self.json_mode = json_mode

    def generate(self, prompt: str) -> str:
        kwargs = {"response_format": {"type": "json_object"}} if self.json_mode else {}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            **kwargs,  # type: ignore
        )

        return cast(str, response.choices[0].message.content)

    def chat(self, chat: List[Dict[str, str]]) -> str:
        kwargs = {"response_format": {"type": "json_object"}} if self.json_mode else {}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=chat,  # type: ignore
            **kwargs,
        )

        return cast(str, response.choices[0].message.content)

    def __call__(self, input: Union[str, List[Dict[str, str]]]) -> str:
        if isinstance(input, str):
            return self.generate(input)
        return self.chat(input)

    def generate_classifier(self, question: str) -> Callable:
        api_doc = CLIP.description + "\n" + str(CLIP.usage)
        prompt = CHOOSE_PARAMS.format(api_doc=api_doc, question=question)
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        params = json.loads(cast(str, response.choices[0].message.content))[
            "Parameters"
        ]

        return lambda x: CLIP()(**{"prompt": params["prompt"], "image": x})

    def generate_detector(self, question: str) -> Callable:
        api_doc = GroundingDINO.description + "\n" + str(GroundingDINO.usage)
        prompt = CHOOSE_PARAMS.format(api_doc=api_doc, question=question)
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        params: Mapping = json.loads(cast(str, response.choices[0].message.content))[
            "Parameters"
        ]

        return lambda x: GroundingDINO()(**{"prompt": params["prompt"], "image": x})

    def generate_segmentor(self, question: str) -> Callable:
        api_doc = GroundingSAM.description + "\n" + str(GroundingSAM.usage)
        prompt = CHOOSE_PARAMS.format(api_doc=api_doc, question=question)
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        params: Mapping = json.loads(cast(str, response.choices[0].message.content))[
            "Parameters"
        ]

        return lambda x: GroundingSAM()(**{"prompt": params["prompt"], "image": x})
