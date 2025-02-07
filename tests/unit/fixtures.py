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
