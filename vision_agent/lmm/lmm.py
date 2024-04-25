import base64
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

import requests
from openai import AzureOpenAI, OpenAI

from vision_agent.tools import CHOOSE_PARAMS, SYSTEM_PROMPT

_LOGGER = logging.getLogger(__name__)

_LLAVA_ENDPOINT = "https://svtswgdnleslqcsjvilau4p6u40jwrkn.lambda-url.us-east-2.on.aws"


def encode_image(image: Union[str, Path]) -> str:
    with open(image, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")
    return encoded_image


class LMM(ABC):
    @abstractmethod
    def generate(
        self, prompt: str, images: Optional[List[Union[str, Path]]] = None
    ) -> str:
        pass

    @abstractmethod
    def chat(
        self,
        chat: List[Dict[str, str]],
        images: Optional[List[Union[str, Path]]] = None,
    ) -> str:
        pass

    @abstractmethod
    def __call__(
        self,
        input: Union[str, List[Dict[str, str]]],
        images: Optional[List[Union[str, Path]]] = None,
    ) -> str:
        pass


class LLaVALMM(LMM):
    r"""An LMM class for the LLaVA-1.6 34B model."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def __call__(
        self,
        input: Union[str, List[Dict[str, str]]],
        images: Optional[List[Union[str, Path]]] = None,
    ) -> str:
        if isinstance(input, str):
            return self.generate(input, images)
        return self.chat(input, images)

    def chat(
        self,
        chat: List[Dict[str, str]],
        images: Optional[List[Union[str, Path]]] = None,
    ) -> str:
        raise NotImplementedError("Chat not supported for LLaVA")

    def generate(
        self,
        prompt: str,
        images: Optional[List[Union[str, Path]]] = None,
        temperature: float = 0.1,
        max_new_tokens: int = 1500,
    ) -> str:
        data = {"prompt": prompt}
        if images and len(images) > 0:
            data["image"] = encode_image(images[0])
        data["temperature"] = temperature  # type: ignore
        data["max_new_tokens"] = max_new_tokens  # type: ignore
        res = requests.post(
            _LLAVA_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=data,
        )
        resp_json: Dict[str, Any] = res.json()
        if (
            "statusCode" in resp_json and resp_json["statusCode"] != 200
        ) or "statusCode" not in resp_json:
            _LOGGER.error(f"Request failed: {resp_json}")
            raise ValueError(f"Request failed: {resp_json}")
        return cast(str, resp_json["data"])


class OpenAILMM(LMM):
    r"""An LMM class for the OpenAI GPT-4 Vision model."""

    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        json_mode: bool = False,
        **kwargs: Any,
    ):
        if not api_key:
            self.client = OpenAI()
        else:
            self.client = OpenAI(api_key=api_key)

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        self.kwargs = kwargs

    def __call__(
        self,
        input: Union[str, List[Dict[str, str]]],
        images: Optional[List[Union[str, Path]]] = None,
    ) -> str:
        if isinstance(input, str):
            return self.generate(input, images)
        return self.chat(input, images)

    def chat(
        self,
        chat: List[Dict[str, str]],
        images: Optional[List[Union[str, Path]]] = None,
    ) -> str:
        fixed_chat = []
        for c in chat:
            fixed_c = {"role": c["role"]}
            fixed_c["content"] = [{"type": "text", "text": c["content"]}]  # type: ignore
            fixed_chat.append(fixed_c)

        if images and len(images) > 0:
            for image in images:
                extension = Path(image).suffix
                if extension.lower() == ".jpeg" or extension.lower() == ".jpg":
                    extension = "jpg"
                elif extension.lower() == ".png":
                    extension = "png"
                else:
                    raise ValueError(f"Unsupported image extension: {extension}")

                encoded_image = encode_image(image)
                fixed_chat[0]["content"].append(  # type: ignore
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{extension};base64,{encoded_image}",
                            "detail": "low",
                        },
                    },
                )

        response = self.client.chat.completions.create(
            model=self.model_name, messages=fixed_chat, **self.kwargs  # type: ignore
        )

        return cast(str, response.choices[0].message.content)

    def generate(
        self,
        prompt: str,
        images: Optional[List[Union[str, Path]]] = None,
    ) -> str:
        message: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if images and len(images) > 0:
            for image in images:
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
            model=self.model_name, messages=message, **self.kwargs  # type: ignore
        )
        return cast(str, response.choices[0].message.content)

    def generate_classifier(self, question: str) -> Callable:
        from vision_agent.tools import CLIP

        api_doc = CLIP.description + "\n" + str(CLIP.usage)
        prompt = CHOOSE_PARAMS.format(api_doc=api_doc, question=question)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        try:
            params = json.loads(cast(str, response.choices[0].message.content))[
                "Parameters"
            ]
        except json.JSONDecodeError:
            _LOGGER.error(
                f"Failed to decode response: {response.choices[0].message.content}"
            )
            raise ValueError("Failed to decode response")

        return lambda x: CLIP()(**{"prompt": params["prompt"], "image": x})

    def generate_detector(self, question: str) -> Callable:
        from vision_agent.tools import GroundingDINO

        api_doc = GroundingDINO.description + "\n" + str(GroundingDINO.usage)
        prompt = CHOOSE_PARAMS.format(api_doc=api_doc, question=question)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        try:
            params = json.loads(cast(str, response.choices[0].message.content))[
                "Parameters"
            ]
        except json.JSONDecodeError:
            _LOGGER.error(
                f"Failed to decode response: {response.choices[0].message.content}"
            )
            raise ValueError("Failed to decode response")

        return lambda x: GroundingDINO()(**{"prompt": params["prompt"], "image": x})

    def generate_segmentor(self, question: str) -> Callable:
        from vision_agent.tools import GroundingSAM

        api_doc = GroundingSAM.description + "\n" + str(GroundingSAM.usage)
        prompt = CHOOSE_PARAMS.format(api_doc=api_doc, question=question)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        try:
            params = json.loads(cast(str, response.choices[0].message.content))[
                "Parameters"
            ]
        except json.JSONDecodeError:
            _LOGGER.error(
                f"Failed to decode response: {response.choices[0].message.content}"
            )
            raise ValueError("Failed to decode response")

        return lambda x: GroundingSAM()(**{"prompt": params["prompt"], "image": x})

    def generate_zero_shot_counter(self, question: str) -> Callable:
        from vision_agent.tools import ZeroShotCounting

        return lambda x: ZeroShotCounting()(**{"image": x})

    def generate_image_qa_tool(self, question: str) -> Callable:
        from vision_agent.tools import ImageQuestionAnswering

        return lambda x: ImageQuestionAnswering()(**{"prompt": question, "image": x})


class AzureOpenAILMM(OpenAILMM):
    def __init__(
        self,
        model_name: str = "gpt-4-vision-preview",
        api_key: Optional[str] = None,
        api_version: str = "2024-02-01",
        azure_endpoint: Optional[str] = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ):
        if not api_key:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not azure_endpoint:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if not api_key:
            raise ValueError("OpenAI API key is required.")
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required.")

        self.client = AzureOpenAI(
            api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint
        )
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.kwargs = kwargs


def get_lmm(name: str) -> LMM:
    if name == "openai":
        return OpenAILMM(name)
    elif name == "llava":
        return LLaVALMM(name)
    else:
        raise ValueError(f"Unknown LMM: {name}, current support openai, llava")
