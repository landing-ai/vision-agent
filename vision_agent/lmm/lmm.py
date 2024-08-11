import base64
import io
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union, cast

import anthropic
import requests
from anthropic.types import ImageBlockParam, MessageParam, TextBlockParam
from openai import AzureOpenAI, OpenAI
from PIL import Image

import vision_agent.tools as T
from vision_agent.tools.prompts import CHOOSE_PARAMS, SYSTEM_PROMPT

from .types import Message

_LOGGER = logging.getLogger(__name__)


def encode_image_bytes(image: bytes) -> str:
    image = Image.open(io.BytesIO(image)).convert("RGB")  # type: ignore
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # type: ignore
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded_image


def encode_media(media: Union[str, Path]) -> str:
    extension = "png"
    extension = Path(media).suffix
    if extension.lower() not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".mp4",
        ".mov",
    }:
        raise ValueError(f"Unsupported image extension: {extension}")

    image_bytes = b""
    if extension.lower() in {".mp4", ".mov"}:
        frames = T.extract_frames(media)
        image = frames[len(frames) // 2]
        buffer = io.BytesIO()
        Image.fromarray(image[0]).convert("RGB").save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
    else:
        image_bytes = open(media, "rb").read()
    return encode_image_bytes(image_bytes)


class LMM(ABC):
    @abstractmethod
    def generate(
        self, prompt: str, media: Optional[List[Union[str, Path]]] = None, **kwargs: Any
    ) -> Union[str, Iterator[Optional[str]]]:
        pass

    @abstractmethod
    def chat(
        self,
        chat: List[Message],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        pass

    @abstractmethod
    def __call__(
        self,
        input: Union[str, List[Message]],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        pass


class OpenAILMM(LMM):
    r"""An LMM class for the OpenAI GPT-4 Vision model."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
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
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        if isinstance(input, str):
            return self.generate(input, **kwargs)
        return self.chat(input, **kwargs)

    def chat(
        self,
        chat: List[Message],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
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
                for media in c["media"]:
                    encoded_media = encode_media(media)

                    fixed_c["content"].append(  # type: ignore
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_media}",
                                "detail": "low",
                            },
                        },
                    )
            fixed_chat.append(fixed_c)

        # prefers kwargs from second dictionary over first
        tmp_kwargs = self.kwargs | kwargs
        response = self.client.chat.completions.create(
            model=self.model_name, messages=fixed_chat, **tmp_kwargs  # type: ignore
        )
        if "stream" in tmp_kwargs and tmp_kwargs["stream"]:

            def f() -> Iterator[Optional[str]]:
                for chunk in response:
                    chunk_message = chunk.choices[0].delta.content  # type: ignore
                    yield chunk_message

            return f()
        else:
            return cast(str, response.choices[0].message.content)

    def generate(
        self,
        prompt: str,
        media: Optional[List[Union[str, Path]]] = None,
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
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
                encoded_media = encode_media(m)
                message[0]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_media}",
                            "detail": "low",
                        },
                    },
                )

        # prefers kwargs from second dictionary over first
        tmp_kwargs = self.kwargs | kwargs
        response = self.client.chat.completions.create(
            model=self.model_name, messages=message, **tmp_kwargs  # type: ignore
        )
        if "stream" in tmp_kwargs and tmp_kwargs["stream"]:

            def f() -> Iterator[Optional[str]]:
                for chunk in response:
                    chunk_message = chunk.choices[0].delta.content  # type: ignore
                    yield chunk_message

            return f()
        else:
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
            response_format={"type": "json_object"},
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
        api_doc = T.get_tool_documentation([T.owl_v2])
        prompt = CHOOSE_PARAMS.format(api_doc=api_doc, question=question)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
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

        return lambda x: T.owl_v2(params["prompt"], x)

    def generate_segmentor(self, question: str) -> Callable:
        api_doc = T.get_tool_documentation([T.grounding_sam])
        prompt = CHOOSE_PARAMS.format(api_doc=api_doc, question=question)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
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
        max_tokens: int = 4096,
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
        self.kwargs = kwargs

    def __call__(
        self,
        input: Union[str, List[Message]],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        if isinstance(input, str):
            return self.generate(input, **kwargs)
        return self.chat(input, **kwargs)

    def chat(
        self,
        chat: List[Message],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
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
                message["images"] = [encode_media(m) for m in message["media"]]
                del message["media"]
            fixed_chat.append(message)
        url = f"{self.url}/chat"
        model = self.model_name
        messages = fixed_chat
        data = {"model": model, "messages": messages}

        tmp_kwargs = self.kwargs | kwargs
        data.update(tmp_kwargs)
        json_data = json.dumps(data)
        if "stream" in tmp_kwargs and tmp_kwargs["stream"]:

            def f() -> Iterator[Optional[str]]:
                with requests.post(url, data=json_data, stream=True) as stream:
                    if stream.status_code != 200:
                        raise ValueError(
                            f"Request failed with status code {stream.status_code}"
                        )

                    for chunk in stream.iter_content(chunk_size=None):
                        chunk_data = json.loads(chunk)
                        if chunk_data["done"]:
                            yield None
                        else:
                            yield chunk_data["message"]["content"]

            return f()
        else:
            stream = requests.post(url, data=json_data)
            if stream.status_code != 200:
                raise ValueError(
                    f"Request failed with status code {stream.status_code}"
                )
            stream = stream.json()
            return stream["message"]["content"]  # type: ignore

    def generate(
        self,
        prompt: str,
        media: Optional[List[Union[str, Path]]] = None,
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:

        url = f"{self.url}/generate"
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [],
        }

        if media and len(media) > 0:
            for m in media:
                data["images"].append(encode_media(m))  # type: ignore

        tmp_kwargs = self.kwargs | kwargs
        data.update(tmp_kwargs)
        json_data = json.dumps(data)
        if "stream" in tmp_kwargs and tmp_kwargs["stream"]:

            def f() -> Iterator[Optional[str]]:
                with requests.post(url, data=json_data, stream=True) as stream:
                    if stream.status_code != 200:
                        raise ValueError(
                            f"Request failed with status code {stream.status_code}"
                        )

                    for chunk in stream.iter_content(chunk_size=None):
                        chunk_data = json.loads(chunk)
                        if chunk_data["done"]:
                            yield None
                        else:
                            yield chunk_data["response"]

            return f()
        else:
            stream = requests.post(url, data=json_data)

            if stream.status_code != 200:
                raise ValueError(
                    f"Request failed with status code {stream.status_code}"
                )

            stream = stream.json()
            return stream["response"]  # type: ignore


class ClaudeSonnetLMM(LMM):
    r"""An LMM class for Anthropic's Claude Sonnet model."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-3-sonnet-20240229",
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = max_tokens
        self.kwargs = kwargs

    def __call__(
        self,
        input: Union[str, List[Dict[str, Any]]],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        if isinstance(input, str):
            return self.generate(input, **kwargs)
        return self.chat(input, **kwargs)

    def chat(
        self,
        chat: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        messages: List[MessageParam] = []
        for msg in chat:
            content: List[Union[TextBlockParam, ImageBlockParam]] = [
                TextBlockParam(type="text", text=msg["content"])
            ]
            if "media" in msg:
                for media_path in msg["media"]:
                    encoded_media = encode_media(media_path)
                    content.append(
                        ImageBlockParam(
                            type="image",
                            source={
                                "type": "base64",
                                "media_type": "image/png",
                                "data": encoded_media,
                            },
                        )
                    )
            messages.append({"role": msg["role"], "content": content})

        # prefers kwargs from second dictionary over first
        tmp_kwargs = self.kwargs | kwargs
        response = self.client.messages.create(
            model=self.model_name, messages=messages, **tmp_kwargs
        )
        if "stream" in tmp_kwargs and tmp_kwargs["stream"]:

            def f() -> Iterator[Optional[str]]:
                for chunk in response:
                    if (
                        chunk.type == "message_start"
                        or chunk.type == "content_block_start"
                    ):
                        continue
                    elif chunk.type == "content_block_delta":
                        yield chunk.delta.text
                    elif chunk.type == "message_stop":
                        yield None

            return f()
        else:
            return cast(str, response.content[0].text)

    def generate(
        self,
        prompt: str,
        media: Optional[List[Union[str, Path]]] = None,
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        content: List[Union[TextBlockParam, ImageBlockParam]] = [
            TextBlockParam(type="text", text=prompt)
        ]
        if media:
            for m in media:
                encoded_media = encode_media(m)
                content.append(
                    ImageBlockParam(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": encoded_media,
                        },
                    )
                )

        # prefers kwargs from second dictionary over first
        tmp_kwargs = self.kwargs | kwargs
        response = self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            **tmp_kwargs,
        )
        if "stream" in tmp_kwargs and tmp_kwargs["stream"]:

            def f() -> Iterator[Optional[str]]:
                for chunk in response:
                    if (
                        chunk.type == "message_start"
                        or chunk.type == "content_block_start"
                    ):
                        continue
                    elif chunk.type == "content_block_delta":
                        yield chunk.delta.text
                    elif chunk.type == "message_stop":
                        yield None

            return f()
        else:
            return cast(str, response.content[0].text)
