from typing import Callable, List, Optional

from .meta_tools import (
    create_code_artifact,
    edit_code_artifact,
    edit_vision_code,
    generate_vision_code,
    get_tool_descriptions,
    list_artifacts,
    open_code_artifact,
    view_media_artifact,
)
from .planner_tools import judge_od_results
from .prompts import CHOOSE_PARAMS, SYSTEM_PROMPT
from .tools import (
    activity_recognition,
    agentic_object_detection,
    agentic_sam2_instance_segmentation,
    agentic_sam2_video_tracking,
    claude35_text_extraction,
    closest_box_distance,
    closest_mask_distance,
    countgd_object_detection,
    countgd_sam2_instance_segmentation,
    countgd_sam2_video_tracking,
    countgd_sam2_visual_instance_segmentation,
    countgd_visual_object_detection,
    custom_object_detection,
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
    get_tools,
    get_tools_descriptions,
    get_tools_df,
    get_tools_docstring,
    get_utilties_docstring,
    glee_object_detection,
    glee_sam2_instance_segmentation,
    glee_sam2_video_tracking,
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
    qwen25_vl_images_vqa,
    qwen25_vl_video_vqa,
    sam2,
    save_image,
    save_json,
    save_video,
    siglip_classification,
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

        global TOOLS, TOOLS_DF, TOOL_DESCRIPTIONS, TOOL_DOCSTRING, TOOLS_INFO
        from vision_agent.tools.tools import TOOLS

        if tool not in TOOLS:  # type: ignore
            TOOLS.append(tool)  # type: ignore

            globals()[tool.__name__] = tool
            if imports is not None:
                for import_ in imports:
                    __new_tools__.append(import_)
            __new_tools__.append(inspect.getsource(tool))
        return tool

    return decorator
