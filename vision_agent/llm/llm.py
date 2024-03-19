import json
from abc import ABC, abstractmethod
from typing import Dict, List, Mapping, Union, cast

from openai import OpenAI

from vision_agent.tools import (
    CHOOSE_PARAMS,
    CLIP,
    SYSTEM_PROMPT,
    GroundingDINO,
    GroundingSAM,
    ImageTool,
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

    def __init__(self, model_name: str = "gpt-4-turbo-preview", json_mode: bool = False):
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
            **kwargs, # type: ignore
        )

        return cast(str, response.choices[0].message.content)

    def chat(self, chat: List[Dict[str, str]]) -> str:
        kwargs = {"response_format": {"type": "json_object"}} if self.json_mode else {}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=chat,  # type: ignore
            **kwargs, # type: ignore
        )

        return cast(str, response.choices[0].message.content)

    def __call__(self, input: Union[str, List[Dict[str, str]]]) -> str:
        if isinstance(input, str):
            return self.generate(input)
        return self.chat(input)

    def generate_classifier(self, prompt: str) -> ImageTool:
        prompt = CHOOSE_PARAMS.format(api_doc=CLIP.description, question=prompt)
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
        return CLIP(**cast(Mapping, params))

    def generate_detector(self, params: str) -> ImageTool:
        api_doc = GroundingDINO.description + "\n" + str(GroundingDINO.usage)
        params = CHOOSE_PARAMS.format(api_doc=api_doc, question=params)
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": params},
            ],
        )

        params = json.loads(cast(str, response.choices[0].message.content))[
            "Parameters"
        ]
        return GroundingDINO(**cast(Mapping, params))

    def generate_segmentor(self, params: str) -> ImageTool:
        params = CHOOSE_PARAMS.format(api_doc=GroundingSAM.description, question=params)
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": params},
            ],
        )

        params = json.loads(cast(str, response.choices[0].message.content))[
            "Parameters"
        ]
        return GroundingSAM(**cast(Mapping, params))
