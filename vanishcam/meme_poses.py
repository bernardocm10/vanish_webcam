"""The meme gestures, ported from the run3 prototype (since removed) with its thresholds unchanged.

Each pose takes the MediaPipe hands and the normalized face box (x1, y1, x2, y2), or None
when no face was seen lately, and returns (matched, score).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .hand_geometry import FINGERTIPS, WRIST, extended_finger_count, extended_finger_states, hand_features, points_of


def finger_states(landmarks) -> list[bool]:
    return extended_finger_states(points_of(landmarks))


def finger_count(landmarks) -> int:
    return extended_finger_count(points_of(landmarks))


def hands_on_face_pose(hands, face_box):
    """Two upright hands close together in front of the face."""
    if len(hands) < 2 or face_box is None:
        return False, 0.0

    fx1, fy1, fx2, fy2 = face_box
    face_width = max(0.001, fx2 - fx1)
    face_height = max(0.001, fy2 - fy1)
    face_center_x = (fx1 + fx2) / 2

    features = sorted((hand_features(hand) for hand in hands[:2]), key=lambda item: item["center"][0])

    valid_hands = []
    overlap_scores = []
    for hand in features:
        center_x, center_y = hand["center"]
        landmarks = hand["landmarks"]
        center_near_face = (
            fx1 - 0.20 * face_width <= center_x <= fx2 + 0.20 * face_width
            and fy1 + 0.18 * face_height <= center_y <= fy2 + 0.35 * face_height
        )
        # In the reference gesture the fingers are above the wrist.
        vertical_hand = hand["wrist_y"] - hand["fingertip_y"] >= 0.08 * face_height
        points_near_face = sum(
            fx1 - 0.18 * face_width <= landmark.x <= fx2 + 0.18 * face_width
            and fy1 - 0.08 * face_height <= landmark.y <= fy2 + 0.35 * face_height
            for landmark in landmarks
        )
        overlap = points_near_face / len(landmarks)
        open_hand = finger_count(landmarks) >= 3
        valid_hands.append(center_near_face and vertical_hand and overlap >= 0.45 and open_hand)
        overlap_scores.append(overlap)

    left_hand, right_hand = features
    left_x = left_hand["center"][0]
    right_x = right_hand["center"][0]
    horizontal_gap = (right_x - left_x) / face_width

    # The hands must be on both sides of the middle of the face.
    balanced = left_x <= face_center_x + 0.16 * face_width and right_x >= face_center_x - 0.16 * face_width
    close_together = 0.04 <= horizontal_gap <= 0.90

    score = float(np.mean(overlap_scores))
    return all(valid_hands) and balanced and close_together, score


def thumbs_up_pose(hands, face_box):
    """A closed hand with only the thumb out, pointing up."""
    if not hands or face_box is None:
        return False, 0.0

    # Each hand on its own: the gesture must work even when MediaPipe sees a second hand
    # that is not doing it.
    if len(hands) > 1:
        results = [thumbs_up_pose([hand], face_box) for hand in hands]
        return max(results, key=lambda result: (result[0], result[1]))

    fx1, fy1, fx2, fy2 = face_box
    face_width = max(0.001, fx2 - fx1)
    face_height = max(0.001, fy2 - fy1)
    face_center_x = (fx1 + fx2) / 2

    hand = hand_features(hands[0])
    landmarks = hand["landmarks"]
    palm_x, palm_y = hand["center"]

    wrist = np.array([landmarks[WRIST].x, landmarks[WRIST].y])
    thumb_mcp = np.array([landmarks[2].x, landmarks[2].y])
    thumb_ip = np.array([landmarks[3].x, landmarks[3].y])
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    middle_mcp = np.array([landmarks[9].x, landmarks[9].y])
    palm_scale = max(0.001, float(np.linalg.norm(middle_mcp - wrist)))

    thumb_axis = thumb_tip - thumb_mcp
    thumb_length = float(np.linalg.norm(thumb_axis))
    thumb_rise = float(thumb_mcp[1] - thumb_tip[1])
    thumb_points_up = (
        thumb_tip[1] < thumb_ip[1] < thumb_mcp[1]
        and thumb_rise >= 0.48 * palm_scale
        and abs(thumb_axis[0]) <= 0.78 * thumb_rise
    )

    # Both thumb segments must be nearly in line: a closed hand with the thumb folded is
    # not taken for an "OK".
    lower_thumb = thumb_ip - thumb_mcp
    upper_thumb = thumb_tip - thumb_ip
    lower_length = max(0.001, float(np.linalg.norm(lower_thumb)))
    upper_length = max(0.001, float(np.linalg.norm(upper_thumb)))
    thumb_alignment = float(np.dot(lower_thumb, upper_thumb) / (lower_length * upper_length))
    thumb_extended = thumb_length >= 0.62 * palm_scale and thumb_alignment >= 0.55

    # Index, middle, ring and little finger must stay folded in the fist.
    folded_fingers = not any(finger_states(landmarks))

    hx1, hy1, hx2, hy2 = hand["bbox"]
    visible_size = hx2 - hx1 >= 0.20 * face_width and hy2 - hy1 >= 0.38 * face_height
    near_upper_body = (
        abs(palm_x - face_center_x) <= 1.65 * face_width
        and fy1 - 0.55 * face_height <= palm_y <= fy2 + 1.90 * face_height
    )

    direction_score = min(1.0, max(0.0, thumb_rise / (0.85 * palm_scale)))
    alignment_score = min(1.0, max(0.0, (thumb_alignment + 1.0) / 2.0))
    fold_score = 1.0 if folded_fingers else 0.0
    score = float(np.mean([direction_score, alignment_score, fold_score]))
    matched = thumb_points_up and thumb_extended and folded_fingers and visible_size and near_upper_body
    return matched, score


def calm_hand_pose(hands, face_box):
    """An open hand held out in front of the chest, palm to the camera."""
    if len(hands) != 1 or face_box is None:
        return False, 0.0

    fx1, fy1, fx2, fy2 = face_box
    face_width = max(0.001, fx2 - fx1)
    face_height = max(0.001, fy2 - fy1)
    face_center_x = (fx1 + fx2) / 2

    hand = hand_features(hands[0])
    landmarks = hand["landmarks"]
    palm_x, palm_y = hand["center"]
    wrist = np.array([landmarks[WRIST].x, landmarks[WRIST].y])
    fingertip_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in FINGERTIPS], axis=0)
    hand_axis = fingertip_center - wrist

    # The hand is in front of the upper chest with the fingers up, even tilted a little.
    in_front_of_chest = (
        abs(palm_x - face_center_x) <= 1.05 * face_width
        and fy2 - 0.12 * face_height <= palm_y <= fy2 + 1.65 * face_height
    )
    fingers_above_wrist = hand_axis[1] <= -0.10 * face_height and abs(hand_axis[1]) >= 0.42 * abs(hand_axis[0])

    four_fingers_open = all(finger_states(landmarks))

    # The four fingers must point roughly the same way: this leaves out spread hands and
    # other gestures with an open palm.
    finger_directions = []
    for tip_index, mcp_index in ((8, 5), (12, 9), (16, 13), (20, 17)):
        direction = np.array([
            landmarks[tip_index].x - landmarks[mcp_index].x,
            landmarks[tip_index].y - landmarks[mcp_index].y,
        ])
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 0.001:
            return False, 0.0
        finger_directions.append(direction / direction_norm)

    adjacent_alignment = [float(np.dot(first, second)) for first, second in zip(finger_directions, finger_directions[1:])]
    fingers_parallel = min(adjacent_alignment) >= 0.72

    fingertip_xs = [landmarks[i].x for i in FINGERTIPS]
    knuckle_xs = [landmarks[i].x for i in (5, 9, 13, 17)]
    fingertip_span = max(fingertip_xs) - min(fingertip_xs)
    knuckle_span = max(0.001, max(knuckle_xs) - min(knuckle_xs))
    fingers_together = fingertip_span <= 1.75 * knuckle_span

    hx1, hy1, hx2, hy2 = hand["bbox"]
    hand_height = hy2 - hy1
    visible_and_frontal = (
        hx2 - hx1 >= 0.34 * face_width
        and hand_height >= 0.48 * face_height
        and knuckle_span >= 0.18 * face_width
    )

    position_score = max(0.0, 1.0 - abs(palm_x - face_center_x) / (1.05 * face_width))
    alignment_score = max(0.0, min(1.0, float(np.mean(adjacent_alignment))))
    size_score = min(1.0, hand_height / (0.78 * face_height))
    score = float(np.mean([position_score, alignment_score, size_score]))
    matched = (
        in_front_of_chest and fingers_above_wrist and four_fingers_open
        and fingers_parallel and fingers_together and visible_and_frontal
    )
    return matched, score


def hearing_hand_pose(hands, face_box):
    """An open hand beside the face, at ear height."""
    if len(hands) != 1 or face_box is None:
        return False, 0.0

    fx1, fy1, fx2, fy2 = face_box
    face_width = max(0.001, fx2 - fx1)
    face_height = max(0.001, fy2 - fy1)
    hand = hand_features(hands[0])
    landmarks = hand["landmarks"]
    center_x, center_y = hand["center"]
    hx1, hy1, hx2, hy2 = hand["bbox"]

    # The palm must be just past one of the two sides of the face.
    beside_left_ear = fx1 - 0.70 * face_width <= center_x <= fx1 + 0.08 * face_width
    beside_right_ear = fx2 - 0.08 * face_width <= center_x <= fx2 + 0.70 * face_width
    beside_face = beside_left_ear or beside_right_ear
    ear_height = fy1 + 0.12 * face_height <= center_y <= fy1 + 0.88 * face_height

    hand_height = hy2 - hy1
    large_enough = hx2 - hx1 >= 0.26 * face_width and hand_height >= 0.42 * face_height
    fingers_up = hand["wrist_y"] - hand["fingertip_y"] >= 0.03 * face_height
    extended_fingers = finger_count(landmarks)
    open_or_cupped = extended_fingers >= 3

    # At least part of the hand must stay close to the side of the head.
    near_head_points = sum(
        fx1 - 0.75 * face_width <= landmark.x <= fx2 + 0.75 * face_width
        and fy1 - 0.18 * face_height <= landmark.y <= fy2 + 0.30 * face_height
        for landmark in landmarks
    )
    proximity = near_head_points / len(landmarks)

    size_score = min(1.0, hand_height / (0.75 * face_height))
    finger_score = extended_fingers / 4
    score = float(np.mean([proximity, size_score, finger_score]))
    matched = beside_face and ear_height and large_enough and fingers_up and open_or_cupped and proximity >= 0.65
    return matched, score


def hands_behind_head_pose(hands, face_box):
    """Two hands apart at both sides of the head, behind the ears."""
    if len(hands) < 2 or face_box is None:
        return False, 0.0

    fx1, fy1, fx2, fy2 = face_box
    face_width = max(0.001, fx2 - fx1)
    face_height = max(0.001, fy2 - fy1)
    face_center_x = (fx1 + fx2) / 2

    features = sorted((hand_features(hand) for hand in hands[:2]), key=lambda item: item["center"][0])
    left_hand, right_hand = features
    left_x, left_y = left_hand["center"]
    right_x, right_y = right_hand["center"]

    left_at_head_side = (
        fx1 - 0.95 * face_width <= left_x <= fx1 + 0.18 * face_width
        and left_x <= face_center_x - 0.24 * face_width
    )
    right_at_head_side = (
        fx2 - 0.18 * face_width <= right_x <= fx2 + 0.95 * face_width
        and right_x >= face_center_x + 0.24 * face_width
    )
    ear_height = all(fy1 - 0.18 * face_height <= y <= fy2 + 0.16 * face_height for y in (left_y, right_y))
    horizontally_separated = right_x - left_x >= 0.78 * face_width
    vertically_aligned = abs(right_y - left_y) <= 0.48 * face_height

    large_enough = True
    for hand in features:
        hx1, hy1, hx2, hy2 = hand["bbox"]
        if hx2 - hx1 < 0.22 * face_width and hy2 - hy1 < 0.32 * face_height:
            large_enough = False

    separation_score = min(1.0, (right_x - left_x) / (1.20 * face_width))
    alignment_score = max(0.0, 1.0 - abs(right_y - left_y) / face_height)
    side_score = float(left_at_head_side and right_at_head_side)
    score = float(np.mean([separation_score, alignment_score, side_score]))
    matched = (
        left_at_head_side and right_at_head_side and ear_height
        and horizontally_separated and vertically_aligned and large_enough
    )
    return matched, score


def prayer_hands_pose(hands, face_box):
    """Two hands pressed together or interlaced, anywhere in the frame."""
    if len(hands) < 2:
        return False, 0.0

    features = [hand_features(hand) for hand in hands[:2]]
    first, second = features
    first_x, first_y = first["center"]
    second_x, second_y = second["center"]

    ax1, ay1, ax2, ay2 = first["bbox"]
    bx1, by1, bx2, by2 = second["bbox"]
    first_size = max(ax2 - ax1, ay2 - ay1)
    second_size = max(bx2 - bx1, by2 - by1)
    hand_scale = max(0.001, (first_size + second_size) / 2)

    center_distance = float(np.hypot(first_x - second_x, first_y - second_y))
    centers_close = center_distance <= 1.05 * hand_scale

    # The areas of the two hands must touch, or be apart by only a small gap, wherever they
    # are compared to the face.
    horizontal_distance = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    vertical_distance = max(0.0, max(ay1, by1) - min(ay2, by2))
    boxes_touching = horizontal_distance <= 0.18 * hand_scale and vertical_distance <= 0.18 * hand_scale

    # With joined hands several landmarks of the two hands end up very close: this makes
    # the gesture easier without taking it for hands merely side by side.
    first_points = points_of(first["landmarks"]).astype(np.float32)
    second_points = points_of(second["landmarks"]).astype(np.float32)
    distances = np.linalg.norm(first_points[:, np.newaxis, :] - second_points[np.newaxis, :, :], axis=2)
    close_threshold = 0.24 * hand_scale
    close_first = int(np.count_nonzero(distances.min(axis=1) <= close_threshold))
    close_second = int(np.count_nonzero(distances.min(axis=0) <= close_threshold))
    landmarks_intertwined = close_first >= 3 and close_second >= 3
    both_hands_open = all(finger_count(hand["landmarks"]) >= 3 for hand in features)

    closeness_score = max(0.0, 1.0 - center_distance / (1.05 * hand_scale))
    contact_score = min(1.0, (close_first + close_second) / 12)
    score = float(np.mean([closeness_score, contact_score]))
    matched = centers_close and boxes_touching and landmarks_intertwined and not both_hands_open
    return matched, score


def timeout_hands_pose(hands, face_box):
    """Two open hands at a right angle, making a T."""
    if len(hands) < 2 or face_box is None:
        return False, 0.0

    fx1, fy1, fx2, fy2 = face_box
    face_width = max(0.001, fx2 - fx1)
    face_height = max(0.001, fy2 - fy1)
    face_center_x = (fx1 + fx2) / 2
    features = [hand_features(hand) for hand in hands[:2]]

    axes = []
    for hand in features:
        landmarks = hand["landmarks"]
        wrist = np.array([landmarks[WRIST].x, landmarks[WRIST].y])
        fingertip_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in FINGERTIPS], axis=0)
        axes.append(fingertip_center - wrist)

    first_axis, second_axis = axes
    first_horizontal = abs(first_axis[0]) >= 1.25 * abs(first_axis[1])
    second_horizontal = abs(second_axis[0]) >= 1.25 * abs(second_axis[1])
    first_vertical = abs(first_axis[1]) >= 1.25 * abs(first_axis[0])
    second_vertical = abs(second_axis[1]) >= 1.25 * abs(second_axis[0])
    one_horizontal_one_vertical = (first_horizontal and second_vertical) or (second_horizontal and first_vertical)

    first_norm = max(0.001, float(np.linalg.norm(first_axis)))
    second_norm = max(0.001, float(np.linalg.norm(second_axis)))
    cosine = abs(float(np.dot(first_axis, second_axis))) / (first_norm * second_norm)
    perpendicular = cosine <= 0.48

    both_open = all(finger_count(hand["landmarks"]) >= 3 for hand in features)

    ax1, ay1, ax2, ay2 = features[0]["bbox"]
    bx1, by1, bx2, by2 = features[1]["bbox"]
    first_size = max(ax2 - ax1, ay2 - ay1)
    second_size = max(bx2 - bx1, by2 - by1)
    hand_scale = max(0.001, (first_size + second_size) / 2)
    horizontal_distance = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    vertical_distance = max(0.0, max(ay1, by1) - min(ay2, by2))
    hands_touching = horizontal_distance <= 0.20 * hand_scale and vertical_distance <= 0.20 * hand_scale

    mean_x = float(np.mean([hand["center"][0] for hand in features]))
    mean_y = float(np.mean([hand["center"][1] for hand in features]))
    in_front_of_chest = (
        abs(mean_x - face_center_x) <= 1.10 * face_width
        and fy2 - 0.22 * face_height <= mean_y <= fy2 + 1.80 * face_height
    )

    angle_score = max(0.0, 1.0 - cosine)
    orientation_score = float(one_horizontal_one_vertical)
    open_score = min(1.0, sum(finger_count(hand["landmarks"]) for hand in features) / 8)
    score = float(np.mean([angle_score, orientation_score, open_score]))
    matched = one_horizontal_one_vertical and perpendicular and both_open and hands_touching and in_front_of_chest
    return matched, score


def pointing_at_self_pose(hands, face_box):
    """One hand below the face with only the index finger, pointing at the chest."""
    if len(hands) != 1 or face_box is None:
        return False, 0.0

    fx1, fy1, fx2, fy2 = face_box
    face_width = max(0.001, fx2 - fx1)
    face_height = max(0.001, fy2 - fy1)
    face_center_x = (fx1 + fx2) / 2

    hand = hand_features(hands[0])
    landmarks = hand["landmarks"]
    palm_x, palm_y = hand["center"]
    wrist_x = landmarks[WRIST].x
    wrist_y = landmarks[WRIST].y
    index_tip_x = landmarks[8].x
    index_tip_y = landmarks[8].y

    states = finger_states(landmarks)
    index_only = states[0] and not any(states[1:])

    # The index tip must be below the chin and near the middle of the body, with the wrist
    # farther out.
    fingertip_at_chest = (
        abs(index_tip_x - face_center_x) <= 0.58 * face_width
        and fy2 - 0.06 * face_height <= index_tip_y <= fy2 + 1.55 * face_height
    )
    palm_below_face = (
        abs(palm_x - face_center_x) <= 0.95 * face_width
        and fy2 - 0.05 * face_height <= palm_y <= fy2 + 1.70 * face_height
    )
    points_inward = abs(index_tip_x - face_center_x) <= abs(wrist_x - face_center_x) - 0.035 * face_width
    points_upward = wrist_y - index_tip_y >= 0.035 * face_height

    hx1, hy1, hx2, hy2 = hand["bbox"]
    visible_size = hx2 - hx1 >= 0.24 * face_width and hy2 - hy1 >= 0.30 * face_height

    inward_score = max(
        0.0,
        min(1.0, (abs(wrist_x - face_center_x) - abs(index_tip_x - face_center_x)) / (0.35 * face_width)),
    )
    center_score = max(0.0, 1.0 - abs(index_tip_x - face_center_x) / face_width)
    finger_score = 1.0 if index_only else 0.0
    score = float(np.mean([inward_score, center_score, finger_score]))
    matched = index_only and fingertip_at_chest and palm_below_face and points_inward and points_upward and visible_size
    return matched, score


# Precedence when two match: thumbs up first, so it works with a second hand in view.
POSES: dict[str, Callable[[list, tuple | None], tuple[bool, float]]] = {
    "thumbs_up": thumbs_up_pose,
    "hands_behind_head": hands_behind_head_pose,
    "timeout": timeout_hands_pose,
    "calm": calm_hand_pose,
    "hands_on_face": hands_on_face_pose,
    "prayer": prayer_hands_pose,
    "pointing_at_self": pointing_at_self_pose,
    "hearing": hearing_hand_pose,
}
GESTURES: tuple[str, ...] = tuple(POSES)


def match_gesture(hands: list, face: tuple | None, enabled: frozenset[str]) -> str | None:
    """The first enabled gesture, in precedence order, that the hands make."""
    for name in GESTURES:
        if name in enabled and POSES[name](hands, face)[0]:
            return name
    return None
