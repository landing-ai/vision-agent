import numpy as np
from template_match import template_matching_with_rotation

import vision_agent as va
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
    agent = va.agent.VisionAgentCoder(verbosity=2)
    result = agent.chat_with_workflow(
        [
            {
                "role": "user",
                "content": "Can you find the locations of the pid_template.png in pid.png and tell me if any are nearby 'NOTE 5'?",
            }
        ],
        media="pid.png",
    )
