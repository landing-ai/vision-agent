import cv2
import numpy as np
import torch
from torchvision.ops import nms


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2,
    )  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def template_matching_with_rotation(
    main_image: np.ndarray,
    template: np.ndarray,
    max_rotation: int = 360,
    step: int = 90,
    threshold: float = 0.75,
    visualize: bool = False,
) -> dict:
    template_height, template_width = template.shape[:2]

    # Convert images to grayscale
    main_image_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    boxes = []
    scores = []

    for angle in range(0, max_rotation, step):
        # Rotate the template
        rotated_template = rotate_image(template_gray, angle)
        if (
            rotated_template.shape[0] > main_image_gray.shape[0]
            or rotated_template.shape[1] > main_image_gray.shape[1]
        ):
            continue

        # Perform template matching
        result = cv2.matchTemplate(
            main_image_gray,
            rotated_template,
            cv2.TM_CCOEFF_NORMED,
        )

        y_coords, x_coords = np.where(result >= threshold)
        for x, y in zip(x_coords, y_coords):
            boxes.append(
                (x, y, x + rotated_template.shape[1], y + rotated_template.shape[0])
            )
            scores.append(result[y, x])

    if len(boxes) > 0:
        indices = (
            nms(
                torch.tensor(boxes).float(),
                torch.tensor(scores).float(),
                0.2,
            )
            .numpy()
            .tolist()
        )
        boxes = [boxes[i] for i in indices]
        scores = [scores[i] for i in indices]

    if visualize:
        # Draw a rectangle around the best match
        for box in boxes:
            cv2.rectangle(main_image, (box[0], box[1]), (box[2], box[3]), 255, 2)

        # Display the result
        cv2.imshow("Best Match", main_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {"bboxes": boxes, "scores": scores}
