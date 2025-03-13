import numpy as np
from template_match import template_matching_with_rotation

import vision_agent as va
import vision_agent.tools as T
import vision_agent.tools.planner_tools as pt
from vision_agent.models import AgentMessage
from vision_agent.utils.image_utils import get_image_size, normalize_bbox


@va.tools.register_tool(
    imports=[
        "import numpy as np",
        "from vision_agent.utils.image_utils import get_image_size, normalize_bbox",
        "from template_match import template_matching_with_rotation",
    ]
)
def template_match(target_image: np.ndarray, template_image: np.ndarray) -> dict:
    """'template_match' tool that finds the locations of the template image in the
    target image.

    Parameters:
        target_image (np.ndarray): The target image.
        template_image (np.ndarray): The template image.

    Returns:
        dict: A dictionary containing the bounding boxes of the matches.

    Example
    -------
    >>> import cv2
    >>> target_image = cv2.imread("pid.png")
    >>> template_image = cv2.imread("pid_template.png")
    >>> matches = template_match(target_image, template_image)
    """

    image_size = get_image_size(target_image)
    matches = template_matching_with_rotation(target_image, template_image)
    matches["bboxes"] = [normalize_bbox(box, image_size) for box in matches["bboxes"]]
    return matches


if __name__ == "__main__":
    agent = va.agent.VisionAgentCoderV2(verbose=True)
    template = T.load_image("pid_template.png")
    pid = T.load_image("pid.png")
    __import__("ipdb").set_trace()
    pt.get_tool_for_task(
        "Find instances of a template image in a larger image",
        {"template": [template], "image": [pid]},
    )
    result = agent.generate_code(
        [
            AgentMessage(
                role="user",
                content="Can you find the locations of the pid_template.png in pid.png and tell me if any are nearby 'NOTE 5'?",
                media=["pid.png", "pid_template.png"],
            )
        ]
    )
