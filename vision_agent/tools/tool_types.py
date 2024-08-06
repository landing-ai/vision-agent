from typing import List, Tuple

from nptyping import UInt8, NDArray, Shape
from pydantic import BaseModel, ConfigDict


class BboxInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: NDArray[Shape["Height, Width, 3"], UInt8]
    filename: str
    labels: List[str]
    bboxes: List[Tuple[int, int, int, int]]


class BboxInputBase64(BaseModel):
    image: str
    filename: str
    labels: List[str]
    bboxes: List[Tuple[int, int, int, int]]
