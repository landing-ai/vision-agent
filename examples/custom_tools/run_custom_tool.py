from template_match import template_matching_with_rotation

import vision_agent as va
from vision_agent.image_utils import get_image_size, normalize_bbox
from vision_agent.tools import Tool, register_tool


@register_tool
class TemplateMatch(Tool):
    name = "template_match_"
    description = "'template_match_' takes a template image and finds all locations where that template appears in the input image."
    usage = {
        "required_parameters": [
            {"name": "target_image", "type": "str"},
            {"name": "template_image", "type": "str"},
        ],
        "examples": [
            {
                "scenario": "Can you detect the location of the template in the target image? Image name: target.png Reference image: template.png",
                "parameters": {
                    "target_image": "target.png",
                    "template_image": "template.png",
                },
            },
        ],
    }

    def __call__(self, target_image: str, template_image: str) -> dict:
        image_size = get_image_size(target_image)
        matches = template_matching_with_rotation(target_image, template_image)
        matches["bboxes"] = [
            normalize_bbox(box, image_size) for box in matches["bboxes"]
        ]
        return matches


if __name__ == "__main__":
    agent = va.agent.VisionAgent(verbose=True)
    resp, tools = agent.chat_with_workflow(
        [
            {
                "role": "user",
                "content": "Can you find the locations of the pid_template.png in pid.png and tell me if any are nearby 'NOTE 5'?",
            }
        ],
        image="pid.png",
        reference_data={"image": "pid_template.png"},
        visualize_output=True,
    )
