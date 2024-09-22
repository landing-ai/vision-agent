import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast

import anthropic
import requests
from anthropic.types import ImageBlockParam, MessageParam, TextBlockParam
from openai import AzureOpenAI, OpenAI

from vision_agent.utils.image_utils import encode_media

from .types import Message


class LMM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        media: Optional[Sequence[Union[str, Path]]] = None,
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        pass

    @abstractmethod
    def chat(
        self,
        chat: Sequence[Message],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        pass

    @abstractmethod
    def __call__(
        self,
        input: Union[str, Sequence[Message]],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        pass


class OpenAILMM(LMM):
    r"""An LMM class for the OpenAI LMMs."""

    def __init__(
        self,
        model_name: str = "gpt-4o-2024-05-13",
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
        input: Union[str, Sequence[Message]],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        if isinstance(input, str):
            return self.generate(input, **kwargs)
        return self.chat(input, **kwargs)

    def chat(
        self,
        chat: Sequence[Message],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        """Chat with the LMM model.

        Parameters:
            chat (Squence[Dict[str, str]]): A list of dictionaries containing the chat
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
                    encoded_media = encode_media(cast(str, media))

                    fixed_c["content"].append(  # type: ignore
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    encoded_media
                                    if encoded_media.startswith(("http", "https"))
                                    or encoded_media.startswith("data:image/")
                                    else f"data:image/png;base64,{encoded_media}"
                                ),
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
        media: Optional[Sequence[Union[str, Path]]] = None,
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
                            "url": (
                                encoded_media
                                if encoded_media.startswith(("http", "https"))
                                or encoded_media.startswith("data:image/")
                                else f"data:image/png;base64,{encoded_media}"
                            ),
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
        num_ctx: int = 128_000,
        **kwargs: Any,
    ):
        """Initializes the Ollama LMM. kwargs are passed as 'options' to the model.
        More information on options can be found here
        https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values

        Parameters:
            model_name (str): The ollama name of the model.
            base_url (str): The base URL of the Ollama API.
            json_mode (bool): Whether to use JSON mode.
            num_ctx (int): The context length for the model.
            kwargs (Any): Additional options to pass to the model.
        """

        self.url = base_url
        self.model_name = model_name
        self.kwargs = {"options": kwargs}

        if json_mode:
            self.kwargs["format"] = "json"  # type: ignore
        self.kwargs["options"]["num_cxt"] = num_ctx

    def __call__(
        self,
        input: Union[str, Sequence[Message]],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        if isinstance(input, str):
            return self.generate(input, **kwargs)
        return self.chat(input, **kwargs)

    def chat(
        self,
        chat: Sequence[Message],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        """Chat with the LMM model.

        Parameters:
            chat (Sequence[Dict[str, str]]): A list of dictionaries containing the chat
                messages. The messages can be in the format:
                [{"role": "user", "content": "Hello!"}, ...]
                or if it contains media, it should be in the format:
                [{"role": "user", "content": "Hello!", "media": ["image1.jpg", ...]}, ...]
        """
        fixed_chat = []
        for message in chat:
            if "media" in message:
                message["images"] = [
                    encode_media(cast(str, m)) for m in message["media"]
                ]
                del message["media"]
            fixed_chat.append(message)
        url = f"{self.url}/chat"
        model = self.model_name
        messages = fixed_chat
        data: Dict[str, Any] = {"model": model, "messages": messages}

        tmp_kwargs = self.kwargs | kwargs
        data.update(tmp_kwargs)
        if "stream" in tmp_kwargs and tmp_kwargs["stream"]:
            json_data = json.dumps(data)

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
            data["stream"] = False
            json_data = json.dumps(data)
            resp = requests.post(url, data=json_data)

            if resp.status_code != 200:
                raise ValueError(f"Request failed with status code {resp.status_code}")
            resp = resp.json()
            return resp["message"]["content"]  # type: ignore

    def generate(
        self,
        prompt: str,
        media: Optional[Sequence[Union[str, Path]]] = None,
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        url = f"{self.url}/generate"
        data: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [],
        }

        if media and len(media) > 0:
            for m in media:
                data["images"].append(encode_media(m))

        tmp_kwargs = self.kwargs | kwargs
        data.update(tmp_kwargs)
        if "stream" in tmp_kwargs and tmp_kwargs["stream"]:
            json_data = json.dumps(data)

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
            data["stream"] = False
            json_data = json.dumps(data)
            resp = requests.post(url, data=json_data)

            if resp.status_code != 200:
                raise ValueError(f"Request failed with status code {resp.status_code}")

            resp = resp.json()
            return resp["response"]  # type: ignore


class AnthropicLMM(LMM):
    r"""An LMM class for Anthropic's LMMs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-3-5-sonnet-20240620",
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
        input: Union[str, Sequence[Dict[str, Any]]],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        if isinstance(input, str):
            return self.generate(input, **kwargs)
        return self.chat(input, **kwargs)

    def chat(
        self,
        chat: Sequence[Dict[str, Any]],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        messages: List[MessageParam] = []
        for msg in chat:
            content: List[Union[TextBlockParam, ImageBlockParam]] = [
                TextBlockParam(type="text", text=msg["content"])
            ]
            if "media" in msg:
                for media_path in msg["media"]:
                    encoded_media = encode_media(media_path, resize=768)
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
        media: Optional[Sequence[Union[str, Path]]] = None,
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        content: List[Union[TextBlockParam, ImageBlockParam]] = [
            TextBlockParam(type="text", text=prompt)
        ]
        if media:
            for m in media:
                encoded_media = encode_media(m, resize=768)
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
