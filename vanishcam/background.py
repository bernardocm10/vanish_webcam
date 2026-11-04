"""The picture behind the person: a chosen image, or the real room kept up to date."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .i18n import t

BACKGROUND_UPDATE_RATE = 0.2
# Exposure gain measured where the person is absent (needs at least 10% of the image).
EXPOSURE_MIN_FREE_FRACTION = 0.1
EXPOSURE_GAIN_RANGE = (0.5, 2.0)


def load_background_image(path: Path) -> np.ndarray:
    # imdecode also reads paths with accents on Windows, where imread fails.
    try:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        image = None
    if image is None:
        raise RuntimeError(t("background.unreadable", path=path))
    return image


def fit_background(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Cover the whole frame without distortion: scale, then crop the excess around the center."""
    src_height, src_width = image.shape[:2]
    scale = max(width / src_width, height / src_height)
    scaled_width = max(width, round(src_width * scale))
    scaled_height = max(height, round(src_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    scaled = cv2.resize(image, (scaled_width, scaled_height), interpolation=interpolation)
    x = (scaled_width - width) // 2
    y = (scaled_height - height) // 2
    return np.ascontiguousarray(scaled[y : y + height, x : x + width])


def replace_background(frame: np.ndarray, background: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep only the person from the frame and put the chosen picture behind them."""
    alpha = mask[..., None]
    return (frame * alpha + background * (1.0 - alpha)).astype(np.uint8)


def update_background(
    background: np.ndarray,
    frame: np.ndarray,
    mask: np.ndarray,
    rate: float = BACKGROUND_UPDATE_RATE,
) -> np.ndarray:
    """Follow lighting changes without recalibrating and without printing the person into the background.

    The webcam's auto exposure changes the whole image (a person in front of a
    window brightens it): that gain, measured where the person is absent, is
    applied to the whole background, also behind the silhouette. Otherwise the
    covered area stays as it was at calibration and, once invisible, shows up as
    a shadow of the person. The frame is then blended in only away from the
    silhouette: the margin keeps uncertain edges (hands, hair) from being printed.
    """
    height, width = mask.shape
    small_size = (max(1, width // 4), max(1, height // 4))
    occupied = (cv2.resize(mask, small_size, interpolation=cv2.INTER_AREA) > 0.05).astype(np.uint8)
    # A margin of at least 4% of the width around the silhouette.
    grow = 2 * max(1, int(np.ceil(small_size[0] * 0.04))) + 1
    occupied = cv2.dilate(occupied, np.ones((grow, grow), np.uint8))
    hard_free = 1.0 - occupied.astype(np.float32)
    background = apply_exposure_gain(background, frame, hard_free > 0, small_size)
    # The feathering stays entirely outside the margin: inside it the weight is zero.
    free = cv2.GaussianBlur(hard_free, (0, 0), sigmaX=grow / 3.0) * hard_free
    weight = cv2.resize(free, (width, height), interpolation=cv2.INTER_NEAREST) * rate
    return cv2.blendLinear(frame, background, weight, 1.0 - weight)


def apply_exposure_gain(
    background: np.ndarray,
    frame: np.ndarray,
    free: np.ndarray,
    small_size: tuple[int, int],
) -> np.ndarray:
    """Bring the whole background to the current exposure, channel by channel."""
    small_bg = cv2.resize(background, small_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    small_frame = cv2.resize(frame, small_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    # Saturated or nearly black pixels say nothing about the gain.
    valid = free & (small_bg.max(axis=2) < 250) & (small_frame.max(axis=2) < 250) & (small_bg.min(axis=2) > 8)
    if np.count_nonzero(valid) < max(16, valid.size * EXPOSURE_MIN_FREE_FRACTION):
        return background
    gain = small_frame[valid].sum(axis=0) / small_bg[valid].sum(axis=0)
    gain = np.clip(gain, EXPOSURE_GAIN_RANGE[0], EXPOSURE_GAIN_RANGE[1])
    if np.all(np.abs(gain - 1.0) < 0.004):
        return background
    return cv2.convertScaleAbs(background * gain.astype(np.float32).reshape(1, 1, 3))


# A started-without-calibration room: what the person covers is painted from around it
# until the camera sees it. Painted at this share of the size: it is a soft guess anyway.
PAINT_SCALE = 0.25
# Below this share of the frame still unseen, the room counts as learned.
LEARNED_FRACTION = 0.002


def covered_area(mask: np.ndarray) -> np.ndarray:
    """Where the person is, with the same safety margin update_background keeps: bool, full size."""
    height, width = mask.shape
    small_size = (max(1, width // 4), max(1, height // 4))
    occupied = (cv2.resize(mask, small_size, interpolation=cv2.INTER_AREA) > 0.05).astype(np.uint8)
    grow = 2 * max(1, int(np.ceil(small_size[0] * 0.04))) + 1
    occupied = cv2.dilate(occupied, np.ones((grow, grow), np.uint8))
    return cv2.resize(occupied, (width, height), interpolation=cv2.INTER_NEAREST) > 0


def paint_unseen(background: np.ndarray, unseen: np.ndarray) -> np.ndarray:
    """The background with the unseen pixels guessed from what is beside them.

    Each row is interpolated between the room seen to the left and to the right of the
    person: a desk, a shelf or the floor line carry on behind them, where a radial
    inpaint smears them into a fan. Rows with nothing seen are taken from the rows above
    and below. Worked out at PAINT_SCALE of the size: it is a soft guess anyway.
    """
    points = cv2.findNonZero(unseen.astype(np.uint8))
    if points is None:
        return background
    height, width = unseen.shape
    x, y, box_width, box_height = cv2.boundingRect(points)
    # Whole rows of the box: the room beside the person is what the guess is made of.
    pad = max(height, width) // 16
    y1, y2 = max(0, y - pad), min(height, y + box_height + pad)
    region = background[y1:y2]
    hole = unseen[y1:y2]
    small_size = (max(8, round(width * PAINT_SCALE)), max(8, round((y2 - y1) * PAINT_SCALE)))
    small = cv2.resize(region, small_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    small_hole = cv2.dilate(cv2.resize(hole.astype(np.uint8), small_size, interpolation=cv2.INTER_NEAREST),
                            np.ones((3, 3), np.uint8)) > 0
    columns = np.arange(small_size[0], dtype=np.float32)
    seen_rows = ~small_hole.all(axis=1)
    for row in np.nonzero(small_hole.any(axis=1) & seen_rows)[0]:
        known = ~small_hole[row]
        for channel in range(3):
            small[row, small_hole[row], channel] = np.interp(
                columns[small_hole[row]], columns[known], small[row, known, channel])
    if seen_rows.any() and not seen_rows.all():
        # Rows hidden from side to side: from the nearest guessed rows above and below.
        rows = np.arange(small_size[1], dtype=np.float32)
        for column in range(small_size[0]):
            for channel in range(3):
                small[~seen_rows, column, channel] = np.interp(
                    rows[~seen_rows], rows[seen_rows], small[seen_rows, column, channel])
    # Softer across the rows than along them: no visible scanlines.
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=0.6, sigmaY=1.2)
    painted = cv2.resize(np.clip(small, 0, 255).astype(np.uint8), (width, y2 - y1), interpolation=cv2.INTER_LINEAR)
    result = background.copy()
    result[y1:y2][hole] = painted[hole]
    return result


def guess_background(frame: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A room from a single frame with the person in it, and where it is still a guess."""
    unseen = covered_area(mask)
    return paint_unseen(frame, unseen), unseen


def learn_unseen(
    background: np.ndarray, frame: np.ndarray, mask: np.ndarray, unseen: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Take what the camera now sees of the guessed area as it is, and paint the rest again.

    The guess is replaced at once rather than blended in slowly: it is only a guess.
    Returns the background and what is still unseen, None once the room is learned.
    """
    revealed = unseen & ~covered_area(mask)
    if revealed.any():
        background = background.copy()
        background[revealed] = frame[revealed]
        unseen = unseen & ~revealed
        # The guess follows its new surroundings.
        background = paint_unseen(background, unseen)
    if np.count_nonzero(unseen) < unseen.size * LEARNED_FRACTION:
        return background, None
    return background, unseen
