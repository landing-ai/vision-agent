import json
import tempfile

import pytest
from PIL import Image

from vision_agent.lmm.lmm import OllamaLMM, OpenAILMM, GoogleLMM

from .fixtures import (  # noqa: F401
    chat_ollama_lmm_mock,
    generate_ollama_lmm_mock,
    openai_lmm_mock,
    google_lmm_mock,
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


@pytest.mark.parametrize(
    "google_lmm_mock", ["mocked response"], indirect=["google_lmm_mock"]
)
def test_google_generate_with_mock(google_lmm_mock):  # noqa: F811
    temp_image = create_temp_image()
    lmm = GoogleLMM()
    response = lmm.generate("test prompt", media=[temp_image])

    assert response == "mocked response"

    # Verify client was called correctly
    google_lmm_mock.models.generate_content.assert_called_once()
    call_args = google_lmm_mock.models.generate_content.call_args

    # Check model name is passed
    assert call_args.kwargs["model"] == "gemini-2.5-pro"

    # Check content includes the prompt
    assert "test prompt" in call_args.kwargs["contents"]

    # Check that a config was passed
    assert "config" in call_args.kwargs


@pytest.mark.parametrize(
    "google_lmm_mock", ["mocked response"], indirect=["google_lmm_mock"]
)
def test_google_generate_with_mock_stream(google_lmm_mock):  # noqa: F811
    temp_image = create_temp_image()
    lmm = GoogleLMM()
    response = lmm.generate("test prompt", media=[temp_image], stream=True)

    expected_response = ["mocked", "response", None]
    for i, chunk in enumerate(response):
        assert chunk == expected_response[i]

    # Verify client was called correctly
    google_lmm_mock.models.generate_content_stream.assert_called_once()
    call_args = google_lmm_mock.models.generate_content_stream.call_args

    # Check model name is passed
    assert call_args.kwargs["model"] == "gemini-2.5-pro"

    # Check content includes the prompt
    assert "test prompt" in call_args.kwargs["contents"]

    # Check that a config was passed
    assert "config" in call_args.kwargs


@pytest.mark.parametrize(
    "google_lmm_mock", ["mocked response"], indirect=["google_lmm_mock"]
)
def test_google_chat_with_mock(google_lmm_mock):  # noqa: F811
    lmm = GoogleLMM()
    response = lmm.chat([{"role": "user", "content": "test prompt"}])

    assert response == "mocked response"

    # Verify client was called correctly
    google_lmm_mock.models.generate_content.assert_called_once()
    call_args = google_lmm_mock.models.generate_content.call_args

    # Check model name is passed
    assert call_args.kwargs["model"] == "gemini-2.5-pro"

    # Verify contents were processed correctly - would be a list of dicts with text
    assert isinstance(call_args.kwargs["contents"], list)
    assert call_args.kwargs["contents"][0]["text"] == "test prompt"


@pytest.mark.parametrize(
    "google_lmm_mock", ["mocked response"], indirect=["google_lmm_mock"]
)
def test_google_chat_with_mock_stream(google_lmm_mock):  # noqa: F811
    lmm = GoogleLMM()
    response = lmm.chat([{"role": "user", "content": "test prompt"}], stream=True)

    expected_response = ["mocked", "response", None]
    for i, chunk in enumerate(response):
        assert chunk == expected_response[i]

    # Verify client was called correctly
    google_lmm_mock.models.generate_content_stream.assert_called_once()
    call_args = google_lmm_mock.models.generate_content_stream.call_args

    # Check model name is passed
    assert call_args.kwargs["model"] == "gemini-2.5-pro"

    # Verify contents were processed correctly
    assert isinstance(call_args.kwargs["contents"], list)
    assert call_args.kwargs["contents"][0]["text"] == "test prompt"


@pytest.mark.parametrize(
    "google_lmm_mock", ["mocked response"], indirect=["google_lmm_mock"]
)
def test_google_call_with_mock(google_lmm_mock):  # noqa: F811
    lmm = GoogleLMM()

    # Test with string input
    response = lmm("test prompt")
    assert response == "mocked response"

    # Test with chat input
    response = lmm([{"role": "user", "content": "test prompt"}])
    assert response == "mocked response"

    # Verify client was called twice
    assert google_lmm_mock.models.generate_content.call_count == 2


@pytest.mark.parametrize(
    "google_lmm_mock", ["mocked response"], indirect=["google_lmm_mock"]
)
def test_google_call_with_mock_stream(google_lmm_mock):  # noqa: F811
    expected_response = ["mocked", "response", None]
    lmm = GoogleLMM()

    # Test with string input
    response = lmm("test prompt", stream=True)
    for i, chunk in enumerate(response):
        assert chunk == expected_response[i]

    # Test with chat input
    response = lmm([{"role": "user", "content": "test prompt"}], stream=True)
    for i, chunk in enumerate(response):
        assert chunk == expected_response[i]

    # Verify client was called twice
    assert google_lmm_mock.models.generate_content_stream.call_count == 2


@pytest.mark.parametrize(
    "google_lmm_mock", ["mocked response"], indirect=["google_lmm_mock"]
)
def test_google_generation_config(google_lmm_mock):  # noqa: F811
    lmm = GoogleLMM()

    # Test with additional generation parameters
    response = lmm.generate(
        "test prompt", temperature=0.7, max_output_tokens=200, top_k=40, top_p=0.95
    )

    assert response == "mocked response"

    # Verify config contains the parameters
    call_args = google_lmm_mock.models.generate_content.call_args
    config = call_args.kwargs["config"]

    # These parameters should be passed in the config
    assert config.temperature == 0.7
    assert config.max_output_tokens == 200
    assert config.top_k == 40
    assert config.top_p == 0.95
