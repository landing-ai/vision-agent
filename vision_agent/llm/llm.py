import json
from abc import ABC, abstractmethod
from typing import Mapping, cast

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


class OpenAILLM(LLM):
    r"""An LLM class for any OpenAI LLM model."""

    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI()

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        return cast(str, response.choices[0].message.content)

    def generate_classifier(self, prompt: str) -> ImageTool:
        prompt = CHOOSE_PARAMS.format(api_doc=CLIP.doc, question=prompt)
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
        params = CHOOSE_PARAMS.format(api_doc=GroundingDINO.doc, question=params)
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
        params = CHOOSE_PARAMS.format(api_doc=GroundingSAM.doc, question=params)
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
