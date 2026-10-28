"""Person mask: where the person is in each frame."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .models import SEGMENTER_MODEL

# Guided filter regularization (on the 0-255 guide scale): lower = finer edges.
MASK_EDGE_EPS = 40.0


def create_segmenter() -> vision.ImageSegmenter:
    options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    return vision.ImageSegmenter.create_from_options(options)


def person_mask(
    segmenter: vision.ImageSegmenter,
    image: mp.Image,
    timestamp_ms: int,
    width: int,
    height: int,
) -> np.ndarray:
    result = segmenter.segment_for_video(image, timestamp_ms)
    if not result.confidence_masks:
        return np.zeros((height, width), dtype=np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    if hasattr(cv2, "ximgproc"):
        # The guided filter snaps the mask to real edges (hair, headphones) and removes
        # the background halo; at half resolution it costs ~5 ms at 720p.
        half = (max(1, width // 2), max(1, height // 2))
        guide = cv2.resize(image.numpy_view(), half, interpolation=cv2.INTER_AREA)
        small = cv2.resize(raw, half, interpolation=cv2.INTER_LINEAR)
        small = cv2.ximgproc.guidedFilter(guide, small, 6, MASK_EDGE_EPS)
        resized = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
        resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=2.2, sigmaY=2.2)
    return np.clip((resized - 0.12) / 0.68, 0.0, 1.0)


class PersonSegmenter:
    """One segmentation per frame: MediaPipe VIDEO rejects repeated timestamps.

    This way the effect and the background update share the same mask.
    """

    def __init__(self, segmenter: vision.ImageSegmenter):
        self.segmenter = segmenter
        self.last: tuple[int, np.ndarray] | None = None

    def __call__(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        if self.last is not None and self.last[0] == timestamp_ms:
            return self.last[1]
        height, width = frame.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask = person_mask(self.segmenter, image, timestamp_ms, width, height)
        self.last = (timestamp_ms, mask)
        return mask
