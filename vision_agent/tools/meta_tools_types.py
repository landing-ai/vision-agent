from enum import Enum
from typing import List, Tuple

from pydantic import BaseModel


class BboxInput(BaseModel):
    image_path: str
    labels: List[str]
    bboxes: List[Tuple[int, int, int, int]]


class BboxInputBase64(BaseModel):
    image: str
    filename: str
    labels: List[str]
    bboxes: List[Tuple[int, int, int, int]]


class PromptTask(str, Enum):
    """
    Valid task prompts options for the Florencev2 model.
    """

    CAPTION = "<CAPTION>"
    """"""
    CAPTION_TO_PHRASE_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"
    """"""
    OBJECT_DETECTION = "<OD>"
    """"""
