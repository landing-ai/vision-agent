from typing import Callable

from .prompts import CHOOSE_PARAMS, SYSTEM_PROMPT
from .tools import (
    TOOL_DESCRIPTIONS,
    TOOL_DOCSTRING,
    TOOLS,
    TOOLS_DF,
    UTILITIES_DOCSTRING,
    clip,
    closest_box_distance,
    closest_mask_distance,
    extract_frames,
    grounding_dino,
    grounding_sam,
    image_caption,
    image_question_answering,
    load_image,
    ocr,
    overlay_bounding_boxes,
    overlay_heat_map,
    overlay_segmentation_masks,
    save_image,
    save_json,
    visual_prompt_counting,
    zero_shot_counting,
)


def register_tool(tool: Callable) -> Callable:
    from .tools import get_tool_descriptions, get_tool_documentation, get_tools_df

    global TOOLS, TOOLS_DF, TOOL_DESCRIPTIONS, TOOL_DOCSTRING

    TOOLS.append(tool)
    TOOLS_DF = get_tools_df(TOOLS)  # type: ignore
    TOOL_DESCRIPTIONS = get_tool_descriptions(TOOLS)
    TOOL_DOCSTRING = get_tool_documentation(TOOLS)  # type: ignore
    return tool
