from typing import Callable, List, Optional

from .meta_tools import META_TOOL_DOCSTRING, florencev2_fine_tuning
from .prompts import CHOOSE_PARAMS, SYSTEM_PROMPT
from .tools import (
    TOOL_DESCRIPTIONS,
    TOOL_DOCSTRING,
    TOOLS,
    TOOLS_DF,
    UTILITIES_DOCSTRING,
    blip_image_caption,
    clip,
    closest_box_distance,
    closest_mask_distance,
    depth_anything_v2,
    detr_segmentation,
    dpt_hybrid_midas,
    extract_frames,
    florencev2_image_caption,
    florencev2_object_detection,
    florencev2_roberta_vqa,
    florencev2_ocr,
    generate_pose_image,
    generate_soft_edge_image,
    get_tool_documentation,
    git_vqa_v2,
    grounding_dino,
    grounding_sam,
    load_image,
    loca_visual_prompt_counting,
    loca_zero_shot_counting,
    ocr,
    overlay_bounding_boxes,
    overlay_heat_map,
    overlay_segmentation_masks,
    owl_v2,
    save_image,
    save_json,
    save_video,
    template_match,
    vit_image_classification,
    vit_nsfw_classification,
)

__new_tools__ = [
    "import vision_agent as va",
    "from vision_agent.tools import register_tool",
]


def register_tool(imports: Optional[List] = None) -> Callable:
    def decorator(tool: Callable) -> Callable:
        import inspect

        from .tools import get_tool_descriptions, get_tools_df

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
