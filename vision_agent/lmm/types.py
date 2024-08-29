from pathlib import Path
from typing import Dict, Sequence, Union

from vision_agent.utils.execute import Execution

TextOrImage = Union[str, Sequence[Union[str, Path]]]
Message = Dict[str, Union[TextOrImage, Execution]]
