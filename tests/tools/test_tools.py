import os
import tempfile

import numpy as np
from PIL import Image

from vision_agent.tools.tools import BboxIoU, SegIoU


def test_bbox_iou():
    bbox1 = [0, 0, 0.75, 0.75]
    bbox2 = [0.25, 0.25, 1, 1]
    assert BboxIoU()(bbox1, bbox2) == 0.29


def test_seg_iou():
    mask1 = np.zeros((10, 10), dtype=np.uint8)
    mask1[2:4, 2:4] = 255
    mask2 = np.zeros((10, 10), dtype=np.uint8)
    mask2[3:5, 3:5] = 255
    with tempfile.TemporaryDirectory() as tmpdir:
        mask1_path = os.path.join(tmpdir, "mask1.png")
        mask2_path = os.path.join(tmpdir, "mask2.png")
        Image.fromarray(mask1).save(mask1_path)
        Image.fromarray(mask2).save(mask2_path)
        assert SegIoU()(mask1_path, mask2_path) == 0.14
