from typing import List, cast

import numpy as np
from vision_agent.configs import Config
from vision_agent.utils.image_utils import convert_to_b64

from vision_agent.agent.visual_design_patterns import DESIGN_PATTERNS

CONFIG = Config()


def suggestion_impl(prompt: str, medias: List[np.ndarray]) -> str:
    suggester = CONFIG.create_suggester()
    if isinstance(medias, np.ndarray):
        medias = [medias]
    all_media_b64 = [
        "data:image/png;base64," + convert_to_b64(media) for media in medias
    ]
    image_sizes = [media.shape[:2] for media in medias]
    resized = suggester.image_size if hasattr(suggester, "image_size") else 768
    image_size = f"The original image sizes were: {str(image_sizes)} and have been resized to {resized}x{resized}"

    prompt = DESIGN_PATTERNS.format(request=prompt, image_size=image_size)

    response = cast(
        str, suggester.generate(prompt, media=all_media_b64, temperature=1.0)
    )
    return response
