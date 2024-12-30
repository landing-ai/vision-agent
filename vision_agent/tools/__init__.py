from typing import Callable, List, Optional

from .meta_tools import (
    create_code_artifact,
    edit_code_artifact,
    edit_vision_code,
    generate_vision_code,
    get_tool_descriptions,
    list_artifacts,
    object_detection_fine_tuning,
    open_code_artifact,
    use_object_detection_fine_tuning,
    view_media_artifact,
)
from .prompts import CHOOSE_PARAMS, SYSTEM_PROMPT
from .tool_utils import add_bboxes_from_masks, get_tool_descriptions_by_names
from .tools import (
    FUNCTION_TOOLS,
    TOOL_DESCRIPTIONS,
    TOOL_DOCSTRING,
    TOOLS,
    TOOLS_DF,
    TOOLS_INFO,
    UTIL_TOOLS,
    UTILITIES_DOCSTRING,
    claude35_text_extraction,
    closest_box_distance,
    closest_mask_distance,
    countgd_object_detection,
    countgd_sam2_instance_segmentation,
    countgd_sam2_video_tracking,
    countgd_visual_prompt_object_detection,
    depth_anything_v2,
    detr_segmentation,
    document_extraction,
    document_qa,
    extract_frames_and_timestamps,
    florence2_object_detection,
    florence2_ocr,
    florence2_sam2_instance_segmentation,
    florence2_sam2_video_tracking,
    flux_image_inpainting,
    generate_pose_image,
    get_tool_documentation,
    load_image,
    minimum_distance,
    ocr,
    od_sam2_video_tracking,
    overlay_bounding_boxes,
    overlay_heat_map,
    overlay_segmentation_masks,
    owlv2_object_detection,
    owlv2_sam2_instance_segmentation,
    owlv2_sam2_video_tracking,
    qwen2_vl_images_vqa,
    qwen2_vl_video_vqa,
    sam2,
    save_image,
    save_json,
    save_video,
    siglip_classification,
    template_match,
    video_temporal_localization,
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

        from .tools import (  # noqa: F811
            get_tool_descriptions,
            get_tools_df,
            get_tools_info,
        )

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
