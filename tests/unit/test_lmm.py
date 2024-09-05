import json
import tempfile

import pytest
from PIL import Image

from vision_agent.lmm.lmm import OllamaLMM, OpenAILMM

from .fixtures import (  # noqa: F401
    chat_ollama_lmm_mock,
    generate_ollama_lmm_mock,
    openai_lmm_mock,
)


def create_temp_image(image_format="jpeg"):
    temp_file = tempfile.NamedTemporaryFile(suffix=f".{image_format}", delete=False)
    image = Image.new("RGB", (100, 100), color=(255, 0, 0))
    image.save(temp_file, format=image_format)
    temp_file.seek(0)
    return temp_file.name


@pytest.mark.parametrize(
    "openai_lmm_mock", ["mocked response"], indirect=["openai_lmm_mock"]
)
def test_generate_with_mock(openai_lmm_mock):  # noqa: F811
    temp_image = create_temp_image()
    lmm = OpenAILMM()
    response = lmm.generate("test prompt", media=[temp_image])
    assert response == "mocked response"
    assert (
        "image_url"
        in openai_lmm_mock.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ][1]
    )


@pytest.mark.parametrize(
    "openai_lmm_mock", ["mocked response"], indirect=["openai_lmm_mock"]
)
def test_generate_with_mock_stream(openai_lmm_mock):  # noqa: F811
    temp_image = create_temp_image()
    lmm = OpenAILMM()
    response = lmm.generate("test prompt", media=[temp_image], stream=True)
    expected_response = ["mocked", "response", None]
    for i, chunk in enumerate(response):
        assert chunk == expected_response[i]
    assert (
        "image_url"
        in openai_lmm_mock.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ][1]
    )


@pytest.mark.parametrize(
    "openai_lmm_mock", ["mocked response"], indirect=["openai_lmm_mock"]
)
def test_chat_with_mock(openai_lmm_mock):  # noqa: F811
    lmm = OpenAILMM()
    response = lmm.chat([{"role": "user", "content": "test prompt"}])
    assert response == "mocked response"
    assert (
        openai_lmm_mock.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ][0]["text"]
        == "test prompt"
    )


@pytest.mark.parametrize(
    "openai_lmm_mock", ["mocked response"], indirect=["openai_lmm_mock"]
)
def test_chat_with_mock_stream(openai_lmm_mock):  # noqa: F811
    lmm = OpenAILMM()
    response = lmm.chat([{"role": "user", "content": "test prompt"}], stream=True)
    expected_response = ["mocked", "response", None]
    for i, chunk in enumerate(response):
        assert chunk == expected_response[i]
    assert (
        openai_lmm_mock.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ][0]["text"]
        == "test prompt"
    )


@pytest.mark.parametrize(
    "openai_lmm_mock", ["mocked response"], indirect=["openai_lmm_mock"]
)
def test_call_with_mock(openai_lmm_mock):  # noqa: F811
    lmm = OpenAILMM()
    response = lmm("test prompt")
    assert response == "mocked response"
    assert (
        openai_lmm_mock.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ][0]["text"]
        == "test prompt"
    )

    response = lmm([{"role": "user", "content": "test prompt"}])
    assert response == "mocked response"
    assert (
        openai_lmm_mock.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ][0]["text"]
        == "test prompt"
    )


@pytest.mark.parametrize(
    "openai_lmm_mock", ["mocked response"], indirect=["openai_lmm_mock"]
)
def test_call_with_mock_stream(openai_lmm_mock):  # noqa: F811
    expected_response = ["mocked", "response", None]
    lmm = OpenAILMM()
    response = lmm("test prompt", stream=True)
    for i, chunk in enumerate(response):
        assert chunk == expected_response[i]
    assert (
        openai_lmm_mock.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ][0]["text"]
        == "test prompt"
    )

    response = lmm([{"role": "user", "content": "test prompt"}], stream=True)
    for i, chunk in enumerate(response):
        assert chunk == expected_response[i]
    assert (
        openai_lmm_mock.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ][0]["text"]
        == "test prompt"
    )


@pytest.mark.parametrize(
    "generate_ollama_lmm_mock",
    ["mocked response"],
    indirect=["generate_ollama_lmm_mock"],
)
def test_generate_ollama_mock(generate_ollama_lmm_mock):  # noqa: F811
    temp_image = create_temp_image()
    lmm = OllamaLMM()
    response = lmm.generate("test prompt", media=[temp_image])
    assert response == "mocked response"
    call_args = json.loads(generate_ollama_lmm_mock.call_args.kwargs["data"])
    assert call_args["prompt"] == "test prompt"


@pytest.mark.parametrize(
    "chat_ollama_lmm_mock", ["mocked response"], indirect=["chat_ollama_lmm_mock"]
)
def test_chat_ollama_mock(chat_ollama_lmm_mock):  # noqa: F811
    lmm = OllamaLMM()
    response = lmm.chat([{"role": "user", "content": "test prompt"}])
    assert response == "mocked response"
    call_args = json.loads(chat_ollama_lmm_mock.call_args.kwargs["data"])
    assert call_args["messages"][0]["content"] == "test prompt"
