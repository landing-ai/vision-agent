from unittest.mock import patch

import numpy as np
import pytest

from vision_agent.llm.llm import OpenAILLM

from .fixtures import langsmith_wrap_oepnai_mock, openai_llm_mock  # noqa: F401


@pytest.mark.parametrize(
    "openai_llm_mock", ["mocked response"], indirect=["openai_llm_mock"]
)
def test_generate_with_mock(openai_llm_mock, langsmith_wrap_oepnai_mock):  # noqa: F811
    llm = OpenAILLM()
    response = llm.generate("test prompt")
    assert response == "mocked response"
    openai_llm_mock.chat.completions.create.assert_called_once_with(
        model="gpt-4o",
        messages=[{"role": "user", "content": "test prompt"}],
    )


@pytest.mark.parametrize(
    "openai_llm_mock", ["mocked response"], indirect=["openai_llm_mock"]
)
def test_chat_with_mock(openai_llm_mock, langsmith_wrap_oepnai_mock):  # noqa: F811
    llm = OpenAILLM()
    response = llm.chat([{"role": "user", "content": "test prompt"}])
    assert response == "mocked response"
    openai_llm_mock.chat.completions.create.assert_called_once_with(
        model="gpt-4o",
        messages=[{"role": "user", "content": "test prompt"}],
    )


@pytest.mark.parametrize(
    "openai_llm_mock_turbo", ["mocked response"], indirect=["openai_llm_mock_turbo"]
)
def openai_llm_mock_turbo(openai_llm_mock_2):  # noqa: F811
    llm = OpenAILLM()
    response = llm.chat([{"role": "user", "content": "test prompt"}])
    assert response == "mocked response"
    openai_llm_mock_turbo.chat.completions.create.assert_called_once_with(
        model="gpt-4o",
        messages=[{"role": "user", "content": "test prompt"}],
    )


@pytest.mark.parametrize(
    "openai_llm_mock", ["mocked response"], indirect=["openai_llm_mock"]
)
def test_call_with_mock(openai_llm_mock, langsmith_wrap_oepnai_mock):  # noqa: F811
    llm = OpenAILLM()
    response = llm("test prompt")
    assert response == "mocked response"
    openai_llm_mock.chat.completions.create.assert_called_once_with(
        model="gpt-4o",
        messages=[{"role": "user", "content": "test prompt"}],
    )

    response = llm([{"role": "user", "content": "test prompt"}])
    assert response == "mocked response"
    openai_llm_mock.chat.completions.create.assert_called_with(
        model="gpt-4o",
        messages=[{"role": "user", "content": "test prompt"}],
    )


@pytest.mark.parametrize(
    "openai_llm_mock",
    ['{"Parameters": {"prompt": "cat"}}'],
    indirect=["openai_llm_mock"],
)
def test_generate_classifier(openai_llm_mock):  # noqa: F811
    with patch("vision_agent.tools.clip") as clip_mock:
        clip_mock.return_value = "test"
        clip_mock.__name__ = "clip"
        clip_mock.__doc__ = "clip"

        llm = OpenAILLM()
        prompt = "Can you generate a cat classifier?"
        classifier = llm.generate_classifier(prompt)
        dummy_image = np.zeros((10, 10, 3)).astype(np.uint8)
        classifier(dummy_image)
        assert clip_mock.call_args[0][1] == "cat"


@pytest.mark.parametrize(
    "openai_llm_mock",
    ['{"Parameters": {"prompt": "cat"}}'],
    indirect=["openai_llm_mock"],
)
def test_generate_detector(openai_llm_mock):  # noqa: F811
    with patch("vision_agent.tools.grounding_dino") as grounding_dino_mock:
        grounding_dino_mock.return_value = "test"
        grounding_dino_mock.__name__ = "grounding_dino"
        grounding_dino_mock.__doc__ = "grounding_dino"

        llm = OpenAILLM()
        prompt = "Can you generate a cat detector?"
        detector = llm.generate_detector(prompt)
        dummy_image = np.zeros((10, 10, 3)).astype(np.uint8)
        detector(dummy_image)
        assert grounding_dino_mock.call_args[0][0] == "cat"


@pytest.mark.parametrize(
    "openai_llm_mock",
    ['{"Parameters": {"prompt": "cat"}}'],
    indirect=["openai_llm_mock"],
)
def test_generate_segmentor(openai_llm_mock):  # noqa: F811
    with patch("vision_agent.tools.grounding_sam") as grounding_sam_mock:
        grounding_sam_mock.return_value = "test"
        grounding_sam_mock.__name__ = "grounding_sam"
        grounding_sam_mock.__doc__ = "grounding_sam"

        llm = OpenAILLM()
        prompt = "Can you generate a cat segmentor?"
        segmentor = llm.generate_segmentor(prompt)
        dummy_image = np.zeros((10, 10, 3)).astype(np.uint8)
        segmentor(dummy_image)
        assert grounding_sam_mock.call_args[0][0] == "cat"
