from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def openai_llm_mock(request):
    content = request.param
    # Note the path here is adjusted to where OpenAI is used, not where it's defined
    with patch("vision_agent.llm.llm.OpenAI") as mock:
        # Setup a mock response structure that matches what your code expects
        mock_instance = mock.return_value
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=content))]
        )
        yield mock_instance


@pytest.fixture
def langsmith_wrap_oepnai_mock(request, openai_llm_mock):
    with patch("vision_agent.llm.llm.wrap_openai") as mock:
        mock.return_value = openai_llm_mock
        yield mock


@pytest.fixture
def openai_lmm_mock(request):
    content = request.param
    # Note the path here is adjusted to where OpenAI is used, not where it's defined
    with patch("vision_agent.lmm.lmm.OpenAI") as mock:
        # Setup a mock response structure that matches what your code expects
        mock_instance = mock.return_value
        mock_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=content))]
        )
        yield mock_instance
