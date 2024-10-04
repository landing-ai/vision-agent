from typing import Callable, List, Optional

from .meta_tools import META_TOOL_DOCSTRING, Artifacts
from .prompts import CHOOSE_PARAMS, SYSTEM_PROMPT
from .tool_utils import get_tool_descriptions_by_names
from .tools import (
    FUNCTION_TOOLS,
    TOOL_DESCRIPTIONS,
    TOOL_DOCSTRING,
    TOOLS,
    TOOLS_DF,
    TOOLS_INFO,
    UTIL_TOOLS,
    UTILITIES_DOCSTRING,
    blip_image_caption,
    clip,
    closest_box_distance,
    closest_mask_distance,
    countgd_counting,
    countgd_example_based_counting,
    depth_anything_v2,
    detr_segmentation,
    dpt_hybrid_midas,
    extract_frames_and_timestamps,
    florence2_image_caption,
    florence2_ocr,
    florence2_phrase_grounding,
    florence2_phrase_grounding_video,
    florence2_roberta_vqa,
    florence2_sam2_image,
    florence2_sam2_video_tracking,
    generate_pose_image,
    generate_soft_edge_image,
    get_tool_documentation,
    git_vqa_v2,
    gpt4o_image_vqa,
    gpt4o_video_vqa,
    grounding_dino,
    grounding_sam,
    ixc25_image_vqa,
    ixc25_temporal_localization,
    ixc25_video_vqa,
    load_image,
    loca_visual_prompt_counting,
    loca_zero_shot_counting,
    ocr,
    overlay_bounding_boxes,
    overlay_heat_map,
    overlay_segmentation_masks,
    owl_v2_image,
    owl_v2_video,
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

        from .tools import get_tool_descriptions, get_tools_df, get_tools_info

        global TOOLS, TOOLS_DF, TOOL_DESCRIPTIONS, TOOL_DOCSTRING, TOOLS_INFO

        if tool not in TOOLS:
            TOOLS.append(tool)
            TOOLS_DF = get_tools_df(TOOLS)  # type: ignore
            TOOL_DESCRIPTIONS = get_tool_descriptions(TOOLS)  # type: ignore
            TOOL_DOCSTRING = get_tool_documentation(TOOLS)  # type: ignore
            TOOLS_INFO = get_tools_info(TOOLS)  # type: ignore

            globals()[tool.__name__] = tool
            if imports is not None:
                for import_ in imports:
                    __new_tools__.append(import_)
            __new_tools__.append(inspect.getsource(tool))
        return tool

    return decorator
