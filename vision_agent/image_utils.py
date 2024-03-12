import base64
from io import BytesIO
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image
from PIL.Image import Image as ImageType


def b64_to_pil(b64_str: str) -> ImageType:
    # , can't be encoded in b64 data so must be part of prefix
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(b64_str)))


def get_image_size(data: Union[str, Path, np.ndarray, ImageType]) -> Tuple[int, ...]:
    if isinstance(data, (str, Path)):
        data = Image.open(data)

    if isinstance(data, Image.Image):
        return data.size[::-1]
    else:
        return data.shape[:2]


def convert_to_b64(data: Union[str, Path, np.ndarray, ImageType]) -> str:
    if data is None:
        raise ValueError(f"Invalid input image: {data}. Input image can't be None.")
    if isinstance(data, (str, Path)):
        data = Image.open(data)
    if isinstance(data, Image.Image):
        buffer = BytesIO()
        data.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        arr_bytes = data.tobytes()
        return base64.b64encode(arr_bytes).decode("utf-8")
