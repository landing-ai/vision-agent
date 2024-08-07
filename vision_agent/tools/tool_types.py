from enum import Enum
from typing import List, Tuple, Literal

from nptyping import UInt8, NDArray
from pydantic import BaseModel, ConfigDict


class BboxInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: NDArray[Literal["Height, Width, 3"], UInt8]
    filename: str
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
