import pytest

from vision_agent.llm.llm import OpenAILLM

from .fixtures import (  # noqa: F401
    clip_mock,
    grounding_dino_mock,
    grounding_sam_mock,
    openai_llm_mock,
)


@pytest.mark.parametrize(
    "openai_llm_mock", ["mocked response"], indirect=["openai_llm_mock"]
)
def test_generate_with_mock(openai_llm_mock):  # noqa: F811
    llm = OpenAILLM()
    response = llm.generate("test prompt")
    assert response == "mocked response"
    openai_llm_mock.chat.completions.create.assert_called_once_with(
        model="gpt-4-turbo",
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
        model="gpt-4-turbo",
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
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": "test prompt"}],
    )

    response = llm([{"role": "user", "content": "test prompt"}])
    assert response == "mocked response"
    openai_llm_mock.chat.completions.create.assert_called_with(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": "test prompt"}],
    )


@pytest.mark.parametrize(
    "openai_llm_mock",
    ['{"Parameters": {"prompt": "cat"}}'],
    indirect=["openai_llm_mock"],
)
def test_generate_classifier(openai_llm_mock, clip_mock):  # noqa: F811
    llm = OpenAILLM()
    prompt = "Can you generate a cat classifier?"
    classifier = llm.generate_classifier(prompt)
    classifier("image.png")
    assert clip_mock.call_args[1] == {"prompt": "cat", "image": "image.png"}


@pytest.mark.parametrize(
    "openai_llm_mock",
    ['{"Parameters": {"prompt": "cat"}}'],
    indirect=["openai_llm_mock"],
)
def test_generate_detector(openai_llm_mock, grounding_dino_mock):  # noqa: F811
    llm = OpenAILLM()
    prompt = "Can you generate a cat detector?"
    detector = llm.generate_detector(prompt)
    detector("image.png")
    assert grounding_dino_mock.call_args[1] == {"prompt": "cat", "image": "image.png"}


@pytest.mark.parametrize(
    "openai_llm_mock",
    ['{"Parameters": {"prompt": "cat"}}'],
    indirect=["openai_llm_mock"],
)
def test_generate_segmentor(openai_llm_mock, grounding_sam_mock):  # noqa: F811
    llm = OpenAILLM()
    prompt = "Can you generate a cat segmentor?"
    segmentor = llm.generate_segmentor(prompt)
    segmentor("image.png")
    assert grounding_sam_mock.call_args[1] == {"prompt": "cat", "image": "image.png"}
