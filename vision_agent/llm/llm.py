import json
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional, Union, cast

from langsmith.wrappers import wrap_openai
from openai import AzureOpenAI, OpenAI

from vision_agent.tools.easytool_tools import (
    CLIP,
    GroundingDINO,
    GroundingSAM,
    ZeroShotCounting,
)
from vision_agent.tools.prompts import CHOOSE_PARAMS, SYSTEM_PROMPT


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
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        json_mode: bool = False,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ):
        if not api_key:
            self.client = wrap_openai(OpenAI())
        else:
            self.client = wrap_openai(OpenAI(api_key=api_key))

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.kwargs = kwargs
        if json_mode:
            self.kwargs["response_format"] = {"type": "json_object"}

    def generate(self, prompt: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore
            **self.kwargs,
        )

        return cast(str, response.choices[0].message.content)

    def chat(self, chat: List[Dict[str, str]]) -> str:
        if self.system_prompt and not any(msg["role"] == "system" for msg in chat):
            chat.insert(0, {"role": "system", "content": self.system_prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=chat,  # type: ignore
            **self.kwargs,
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

    def generate_zero_shot_counter(self, question: str) -> Callable:
        return lambda x: ZeroShotCounting()(**{"image": x})

    def generate_image_qa_tool(self, question: str) -> Callable:
        from vision_agent.tools.easytool_tools import ImageQuestionAnswering

        return lambda x: ImageQuestionAnswering()(**{"prompt": question, "image": x})


class AzureOpenAILLM(OpenAILLM):
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_version: str = "2024-02-01",
        azure_endpoint: Optional[str] = None,
        json_mode: bool = False,
        **kwargs: Any
    ):
        if not api_key:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not azure_endpoint:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if not api_key:
            raise ValueError("Azure OpenAI API key is required.")
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required.")

        self.client = wrap_openai(
            AzureOpenAI(
                api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint
            )
        )
        self.model_name = model_name
        self.kwargs = kwargs
        if json_mode:
            self.kwargs["response_format"] = {"type": "json_object"}
