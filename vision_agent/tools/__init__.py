from typing import Callable, List, Optional

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
    save_video,
    visual_prompt_counting,
    zero_shot_counting,
)

__new_tools__ = [
    "import vision_agent as va",
    "from vision_agent.tools import register_tool",
]


def register_tool(imports: Optional[List] = None) -> Callable:
    def decorator(tool: Callable) -> Callable:
        import inspect

        from .tools import get_tool_descriptions, get_tool_documentation, get_tools_df

        global TOOLS, TOOLS_DF, TOOL_DESCRIPTIONS, TOOL_DOCSTRING

        if tool not in TOOLS:
            TOOLS.append(tool)
            TOOLS_DF = get_tools_df(TOOLS)  # type: ignore
            TOOL_DESCRIPTIONS = get_tool_descriptions(TOOLS)  # type: ignore
            TOOL_DOCSTRING = get_tool_documentation(TOOLS)  # type: ignore

            globals()[tool.__name__] = tool
            if imports is not None:
                for import_ in imports:
                    __new_tools__.append(import_)
            __new_tools__.append(inspect.getsource(tool))
        return tool

    return decorator
