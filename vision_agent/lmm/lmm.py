import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
import base64

import anthropic
import requests
from anthropic.types import ImageBlockParam, MessageParam, TextBlockParam
from openai import AzureOpenAI, OpenAI

from google import genai  # type: ignore
from google.genai import types  # type: ignore

from vision_agent.models import Message
from vision_agent.utils.image_utils import encode_media


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
        image_size: int = 768,
        image_detail: str = "low",
        **kwargs: Any,
    ):
        if not api_key:
            self.client = OpenAI()
        else:
            self.client = OpenAI(api_key=api_key)

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.image_size = image_size
        self.image_detail = image_detail
        # o1 does not use max_tokens
        if "max_tokens" not in kwargs and not (
            model_name.startswith("o1") or model_name.startswith("o3")
        ):
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
            if "media" in c and self.model_name != "o3-mini":
                for media in c["media"]:
                    resize = kwargs["resize"] if "resize" in kwargs else self.image_size
                    image_detail = (
                        kwargs["image_detail"]
                        if "image_detail" in kwargs
                        else self.image_detail
                    )
                    encoded_media = encode_media(cast(str, media), resize=resize)

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
                                "detail": image_detail,
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
        if media and len(media) > 0 and self.model_name != "o3-mini":
            for m in media:
                resize = kwargs["resize"] if "resize" in kwargs else None
                image_detail = (
                    kwargs["image_detail"]
                    if "image_detail" in kwargs
                    else self.image_detail
                )
                encoded_media = encode_media(m, resize=resize)
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
                            "detail": image_detail,
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
        image_detail: str = "low",
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
        self.image_detail = image_detail

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
        image_size: int = 768,
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
        self.image_size = image_size
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
                resize = kwargs["resize"] if "resize" in kwargs else self.image_size
                message["images"] = [
                    encode_media(cast(str, m), resize=resize) for m in message["media"]
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
                resize = kwargs["resize"] if "resize" in kwargs else self.image_size
                data["images"].append(encode_media(m, resize=resize))

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
        image_size: int = 768,
        **kwargs: Any,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.image_size = image_size
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
                    resize = kwargs["resize"] if "resize" in kwargs else self.image_size
                    encoded_media = encode_media(media_path, resize=resize)
                    if encoded_media.startswith("data:image/png;base64,"):
                        encoded_media = encoded_media[len("data:image/png;base64,") :]
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
                resize = kwargs["resize"] if "resize" in kwargs else self.image_size
                encoded_media = encode_media(m, resize=resize)
                if encoded_media.startswith("data:image/png;base64,"):
                    encoded_media = encoded_media[len("data:image/png;base64,") :]
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


class GoogleLMM(LMM):
    r"""An LMM class for the Google LMMs."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro-preview-03-25",
        api_key: Optional[str] = None,
        image_size: int = 768,
        image_detail: str = "low",
        **kwargs: Any,
    ):
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")

        # Create the client using the Google Genai client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.image_size = image_size
        self.image_detail = image_detail
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
        prompt_parts = []
        for message in chat:
            if message["role"] != "user":
                continue  # Gemini expects only user input
            prompt_parts.extend(self._convert_message_parts(message, **kwargs))

        tmp_kwargs = self.kwargs | kwargs
        generation_config = self._create_generation_config(tmp_kwargs)

        if tmp_kwargs.get("stream"):

            def f() -> Iterator[Optional[str]]:
                # Use the client to stream content
                response_stream = self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=prompt_parts,
                    config=generation_config,
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text

            return f()
        else:
            # Use the client for non-streaming
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_parts,
                config=generation_config,
            )
            return cast(str, response.text)

    def generate(
        self,
        prompt: str,
        media: Optional[Sequence[Union[str, Path]]] = None,
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        prompt_parts = [{"text": prompt}]
        if media:
            for m in media:
                prompt_parts.append(self._convert_media_part(m, **kwargs))

        tmp_kwargs = self.kwargs | kwargs
        generation_config = self._create_generation_config(tmp_kwargs)

        if tmp_kwargs.get("stream"):

            def f() -> Iterator[Optional[str]]:
                response_stream = self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=prompt_parts,
                    config=generation_config,
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text

            return f()
        else:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_parts,
                config=generation_config,
            )
            return cast(str, response.text)

    def _convert_message_parts(
        self, message: Dict[str, Any], **kwargs: Any
    ) -> List[Any]:
        parts = [{"text": message["content"]}]
        if "media" in message:
            for media_path in message["media"]:
                parts.append(self._convert_media_part(media_path, **kwargs))
        return parts

    def _convert_media_part(self, media: Union[str, Path], **kwargs: Any) -> types.Part:
        resize = kwargs.get("resize", self.image_size)
        encoded_media = encode_media(str(media), resize=resize)

        if encoded_media.startswith("data:image/"):
            encoded_media = encoded_media.split(",", 1)[-1]

        binary_data = base64.b64decode(encoded_media)

        return types.Part.from_bytes(
            data=binary_data,
            mime_type="image/png",
        )

    def _create_generation_config(
        self, kwargs: Dict[str, Any]
    ) -> types.GenerateContentConfig:
        # Extract generation-specific parameters
        config_params = {}

        # Handle known parameters
        for param in [
            "max_output_tokens",
            "temperature",
            "top_p",
            "top_k",
            "response_mime_type",
            "stop_sequences",
            "candidate_count",
            "seed",
            "safety_settings",
            "system_instruction",
        ]:
            if param in kwargs:
                config_params[param] = kwargs[param]

        # Create a GenerateContentConfig object
        return types.GenerateContentConfig(**config_params)
