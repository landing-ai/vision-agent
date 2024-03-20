from unittest.mock import MagicMock, patch

import pytest

from vision_agent.tools import CLIP, GroundingDINO, GroundingSAM


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


@pytest.fixture
def clip_mock(request):
    with patch.object(CLIP, "__call__", autospec=True) as mock:
        mock.return_value = "test"
        yield mock


@pytest.fixture
def grounding_dino_mock(request):
    with patch.object(GroundingDINO, "__call__", autospec=True) as mock:
        mock.return_value = "test"
        yield mock


@pytest.fixture
def grounding_sam_mock(request):
    with patch.object(GroundingSAM, "__call__", autospec=True) as mock:
        mock.return_value = "test"
        yield mock
