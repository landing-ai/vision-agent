import pytest

from vision_agent.llm.llm import OpenAILLM
from vision_agent.tools import CLIP, GroundingDINO, GroundingSAM

from .fixtures import openai_llm_mock  # noqa: F401


@pytest.mark.parametrize(
    "openai_llm_mock", ["mocked response"], indirect=["openai_llm_mock"]
)
def test_generate_with_mock(openai_llm_mock):  # noqa: F811
    llm = OpenAILLM()
    response = llm.generate("test prompt")
    assert response == "mocked response"
    openai_llm_mock.chat.completions.create.assert_called_once_with(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": "test prompt"}],
    )


@pytest.mark.parametrize(
    "openai_llm_mock", ["mocked response"], indirect=["openai_llm_mock"]
)
def test_chat_with_mock(openai_llm_mock):  # noqa: F811
    llm = OpenAILLM()
    response = llm.chat([{"role": "user", "content": "test prompt"}])
    assert response == "mocked response"
    openai_llm_mock.chat.completions.create.assert_called_once_with(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": "test prompt"}],
    )


@pytest.mark.parametrize(
    "openai_llm_mock", ["mocked response"], indirect=["openai_llm_mock"]
)
def test_call_with_mock(openai_llm_mock):  # noqa: F811
    llm = OpenAILLM()
    response = llm("test prompt")
    assert response == "mocked response"
    openai_llm_mock.chat.completions.create.assert_called_once_with(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": "test prompt"}],
    )

    response = llm([{"role": "user", "content": "test prompt"}])
    assert response == "mocked response"
    openai_llm_mock.chat.completions.create.assert_called_with(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": "test prompt"}],
    )


@pytest.mark.parametrize(
    "openai_llm_mock",
    ['{"Parameters": {"prompt": "cat"}}'],
    indirect=["openai_llm_mock"],
)
def test_generate_classifier(openai_llm_mock):  # noqa: F811
    llm = OpenAILLM()
    prompt = "Can you generate a cat classifier?"
    classifier = llm.generate_classifier(prompt)
    assert isinstance(classifier, CLIP)
    assert classifier.prompt == "cat"


@pytest.mark.parametrize(
    "openai_llm_mock",
    ['{"Parameters": {"prompt": "cat"}}'],
    indirect=["openai_llm_mock"],
)
def test_generate_detector(openai_llm_mock):  # noqa: F811
    llm = OpenAILLM()
    prompt = "Can you generate a cat detector?"
    detector = llm.generate_detector(prompt)
    assert isinstance(detector, GroundingDINO)
    assert detector.prompt == "cat"


@pytest.mark.parametrize(
    "openai_llm_mock",
    ['{"Parameters": {"prompt": "cat"}}'],
    indirect=["openai_llm_mock"],
)
def test_generate_segmentor(openai_llm_mock):  # noqa: F811
    llm = OpenAILLM()
    prompt = "Can you generate a cat segmentor?"
    segmentor = llm.generate_segmentor(prompt)
    assert isinstance(segmentor, GroundingSAM)
    assert segmentor.prompt == "cat"
