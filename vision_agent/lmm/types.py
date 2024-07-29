from pathlib import Path
from typing import Dict, Sequence, Union

TextOrImage = Union[str, Sequence[Union[str, Path]]]
Message = Dict[str, TextOrImage]
