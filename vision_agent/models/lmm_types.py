from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np
from PIL.Image import Image as ImageType

from vision_agent.utils.execute import Execution

TextOrImage = Union[str, Sequence[Union[str, Path, ImageType, np.ndarray]]]
Message = Dict[str, Union[TextOrImage, Execution]]
