from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def openai_lmm_mock(request):
    content = request.param

    def mock_generate(*args, **kwargs):
        if kwargs.get("stream", False):

            def generator():
                for chunk in content.split(" ") + [None]:
                    yield MagicMock(choices=[MagicMock(delta=MagicMock(content=chunk))])

            return generator()
        else:
            return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])

    # Note the path here is adjusted to where OpenAI is used, not where it's defined
    with patch("vision_agent.lmm.lmm.OpenAI") as mock:
        # Setup a mock response structure that matches what your code expects
        mock_instance = mock.return_value
        mock_instance.chat.completions.create.return_value = mock_generate()
        yield mock_instance


@pytest.fixture
def generate_ollama_lmm_mock(request):
    content = request.param

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": content}
    with patch("vision_agent.lmm.lmm.requests.post") as mock:
        mock.return_value = mock_resp
        yield mock


@pytest.fixture
def chat_ollama_lmm_mock(request):
    content = request.param

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": content}}
    with patch("vision_agent.lmm.lmm.requests.post") as mock:
        mock.return_value = mock_resp
        yield mock


@pytest.fixture
def google_lmm_mock(request):
    content = request.param

    # Mock implementation for streaming responses
    def mock_stream_generator():
        for chunk in content.split(" "):
            yield MagicMock(text=chunk)
        yield MagicMock(text=None)

    # Mock implementation for regular responses
    mock_generate_response = MagicMock()
    mock_generate_response.text = content

    # Set up the client mock
    mock_client = MagicMock()
    mock_models = MagicMock()
    mock_client.models = mock_models

    # Configure generate_content method
    mock_models.generate_content.return_value = mock_generate_response

    # Configure generate_content_stream method
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = mock_stream_generator()
    mock_models.generate_content_stream.return_value = mock_stream

    # Patch the genai.Client class
    with patch("google.genai.Client") as mock_client_class:
        mock_client_class.return_value = mock_client
        yield mock_client

@pytest.fixture
def anthropic_lmm_mock(request):
    content = request.param

    def mock_create(*args, **kwargs):
        if kwargs.get("stream", False):

            def generator():
                chunks = []
                for chunk in content.split(" "):
                    chunks.append(
                        MagicMock(
                            type="content_block_delta",
                            delta=MagicMock(text=chunk + " " if chunk else ""),
                        )
                    )
                chunks.append(MagicMock(type="message_stop"))
                for chunk in chunks:
                    yield chunk

            return generator()
        else:
            return MagicMock(content=[MagicMock(text=content, type="text")])

    with patch("vision_agent.lmm.lmm.anthropic.Anthropic") as mock:
        mock_instance = mock.return_value
        mock_instance.messages.create.return_value = mock_create()
        yield mock_instance
