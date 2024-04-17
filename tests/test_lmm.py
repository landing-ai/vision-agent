import tempfile

import pytest
from PIL import Image

from vision_agent.lmm.lmm import OpenAILMM

from .fixtures import (  # noqa: F401
    clip_mock,
    grounding_dino_mock,
    grounding_sam_mock,
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
    response = lmm.generate("test prompt", images=[temp_image])
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
    "openai_lmm_mock",
    ['{"Parameters": {"prompt": "cat"}}'],
    indirect=["openai_lmm_mock"],
)
def test_generate_classifier(openai_lmm_mock, clip_mock):  # noqa: F811
    lmm = OpenAILMM()
    prompt = "Can you generate a cat classifier?"
    classifier = lmm.generate_classifier(prompt)
    classifier("image.png")
    assert clip_mock.call_args[1] == {"prompt": "cat", "image": "image.png"}


@pytest.mark.parametrize(
    "openai_lmm_mock",
    ['{"Parameters": {"prompt": "cat"}}'],
    indirect=["openai_lmm_mock"],
)
def test_generate_detector(openai_lmm_mock, grounding_dino_mock):  # noqa: F811
    lmm = OpenAILMM()
    prompt = "Can you generate a cat classifier?"
    detector = lmm.generate_detector(prompt)
    detector("image.png")
    assert grounding_dino_mock.call_args[1] == {"prompt": "cat", "image": "image.png"}


@pytest.mark.parametrize(
    "openai_lmm_mock",
    ['{"Parameters": {"prompt": "cat"}}'],
    indirect=["openai_lmm_mock"],
)
def test_generate_segmentor(openai_lmm_mock, grounding_sam_mock):  # noqa: F811
    lmm = OpenAILMM()
    prompt = "Can you generate a cat classifier?"
    segmentor = lmm.generate_segmentor(prompt)
    segmentor("image.png")
    assert grounding_sam_mock.call_args[1] == {"prompt": "cat", "image": "image.png"}
