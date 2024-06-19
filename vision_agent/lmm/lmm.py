import base64
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

import requests
from openai import AzureOpenAI, OpenAI

import vision_agent.tools as T
from vision_agent.tools.prompts import CHOOSE_PARAMS, SYSTEM_PROMPT

_LOGGER = logging.getLogger(__name__)


def encode_image(image: Union[str, Path]) -> str:
    with open(image, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")
    return encoded_image


TextOrImage = Union[str, List[Union[str, Path]]]
Message = Dict[str, TextOrImage]


class LMM(ABC):
    @abstractmethod
    def generate(
        self, prompt: str, media: Optional[List[Union[str, Path]]] = None
    ) -> str:
        pass

    @abstractmethod
    def chat(
        self,
        chat: List[Message],
    ) -> str:
        pass

    @abstractmethod
    def __call__(
        self,
        input: Union[str, List[Message]],
    ) -> str:
        pass


class OpenAILMM(LMM):
    r"""An LMM class for the OpenAI GPT-4 Vision model."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
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
        input: Union[str, List[Message]],
    ) -> str:
        if isinstance(input, str):
            return self.generate(input)
        return self.chat(input)

    def chat(
        self,
        chat: List[Message],
    ) -> str:
        """Chat with the LMM model.

        Parameters:
            chat (List[Dict[str, str]]): A list of dictionaries containing the chat
                messages. The messages can be in the format:
                [{"role": "user", "content": "Hello!"}, ...]
                or if it contains media, it should be in the format:
                [{"role": "user", "content": "Hello!", "media": ["image1.jpg", ...]}, ...]
        """
        fixed_chat = []
        for c in chat:
            fixed_c = {"role": c["role"]}
            fixed_c["content"] = [{"type": "text", "text": c["content"]}]  # type: ignore
            if "media" in c:
                for image in c["media"]:
                    extension = Path(image).suffix
                    if extension.lower() == ".jpeg" or extension.lower() == ".jpg":
                        extension = "jpg"
                    elif extension.lower() == ".png":
                        extension = "png"
                    else:
                        raise ValueError(f"Unsupported image extension: {extension}")
                    encoded_image = encode_image(image)
                    fixed_c["content"].append(  # type: ignore
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{extension};base64,{encoded_image}",  # type: ignore
                                "detail": "low",
                            },
                        },
                    )
            fixed_chat.append(fixed_c)

        response = self.client.chat.completions.create(
            model=self.model_name, messages=fixed_chat, **self.kwargs  # type: ignore
        )

        return cast(str, response.choices[0].message.content)

    def generate(
        self,
        prompt: str,
        media: Optional[List[Union[str, Path]]] = None,
    ) -> str:
        message: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if media and len(media) > 0:
            for m in media:
                extension = Path(m).suffix
                encoded_image = encode_image(m)
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
        api_doc = T.get_tool_documentation([T.clip])
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

        return lambda x: T.clip(x, params["prompt"])

    def generate_detector(self, question: str) -> Callable:
        api_doc = T.get_tool_documentation([T.grounding_dino])
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

        return lambda x: T.grounding_dino(params["prompt"], x)

    def generate_segmentor(self, question: str) -> Callable:
        api_doc = T.get_tool_documentation([T.grounding_sam])
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

        return lambda x: T.grounding_sam(params["prompt"], x)

    def generate_zero_shot_counter(self, question: str) -> Callable:
        return T.loca_zero_shot_counting

    def generate_image_qa_tool(self, question: str) -> Callable:
        return lambda x: T.git_vqa_v2(question, x)


class AzureOpenAILMM(OpenAILMM):
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-01",
        azure_endpoint: Optional[str] = None,
        max_tokens: int = 1024,
        json_mode: bool = False,
        **kwargs: Any,
    ):
        if not api_key:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not azure_endpoint:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not model_name:
            model_name = os.getenv("AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT_NAME")

        if not api_key:
            raise ValueError("OpenAI API key is required.")
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required.")
        if not model_name:
            raise ValueError("Azure OpenAI chat model deployment name is required.")

        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )
        self.model_name = model_name

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        self.kwargs = kwargs


class OllamaLMM(LMM):
    r"""An LMM class for the ollama."""

    def __init__(
        self,
        model_name: str = "llava",
        base_url: Optional[str] = "http://localhost:11434/api",
        json_mode: bool = False,
        **kwargs: Any,
    ):
        self.url = base_url
        self.model_name = model_name
        self.json_mode = json_mode
        self.stream = False

    def __call__(
        self,
        input: Union[str, List[Message]],
    ) -> str:
        if isinstance(input, str):
            return self.generate(input)
        return self.chat(input)

    def chat(
        self,
        chat: List[Message],
    ) -> str:
        """Chat with the LMM model.

        Parameters:
            chat (List[Dict[str, str]]): A list of dictionaries containing the chat
                messages. The messages can be in the format:
                [{"role": "user", "content": "Hello!"}, ...]
                or if it contains media, it should be in the format:
                [{"role": "user", "content": "Hello!", "media": ["image1.jpg", ...]}, ...]
        """
        fixed_chat = []
        for message in chat:
            if "media" in message:
                message["images"] = [encode_image(m) for m in message["media"]]
                del message["media"]
            fixed_chat.append(message)
        url = f"{self.url}/chat"
        model = self.model_name
        messages = fixed_chat
        data = {"model": model, "messages": messages, "stream": self.stream}
        json_data = json.dumps(data)
        response = requests.post(url, data=json_data)
        if response.status_code != 200:
            raise ValueError(f"Request failed with status code {response.status_code}")
        response = response.json()
        return response["message"]["content"]  # type: ignore

    def generate(
        self,
        prompt: str,
        media: Optional[List[Union[str, Path]]] = None,
    ) -> str:

        url = f"{self.url}/generate"
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [],
            "stream": self.stream,
        }

        json_data = json.dumps(data)
        if media and len(media) > 0:
            for m in media:
                data["images"].append(encode_image(m))  # type: ignore

        response = requests.post(url, data=json_data)

        if response.status_code != 200:
            raise ValueError(f"Request failed with status code {response.status_code}")

        response = response.json()
        return response["response"]  # type: ignore
