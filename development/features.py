"""
Facial feature engineering.

Each function turns raw landmark points into one numeric measurement.
extract_features() combines them into a single named dict — using names
(not just a bare list of numbers) matters later: it's what lets you map
model coefficients back to "cheekbone_distance" for the goofy score sheet,
instead of losing track of which number means what.
"""

import numpy as np


def _pixel_point(landmark, width, height):
    return np.array([landmark.x * width, landmark.y * height])


def _distance(landmarks, idx_a, idx_b, width, height):
    a = _pixel_point(landmarks[idx_a], width, height)
    b = _pixel_point(landmarks[idx_b], width, height)
    return float(np.linalg.norm(a - b))

def _jaw_angle(landmarks, width, height):
    """Returns the angle of the jawline (landmarks 152, 234, 454) in degrees.

    The jawline is a triangle with points at the chin and both cheekbones.
    This function returns the angle at the chin, which is a good proxy for
    how "pointy" or "square" the jaw is. A smaller angle means a more
    pronounced chin, while a larger angle means a wider jaw.
    """
    a = _pixel_point(landmarks[234], width, height)  # left cheekbone
    b = _pixel_point(landmarks[152], width, height)  # chin
    c = _pixel_point(landmarks[454], width, height)  # right cheekbone

    ab = a - b
    cb = c - b

    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def _cheek_hollowness(landmarks, width, height):
    """Returns a ratio of cheek hollowness.

    This is computed as the distance from the cheekbone to the corner of the
    mouth, divided by the distance from the cheekbone to the chin. A larger
    ratio indicates a more pronounced hollow in the cheek.
    """
    left_cheekbone = _pixel_point(landmarks[234], width, height)
    right_cheekbone = _pixel_point(landmarks[454], width, height)
    left_mouth_corner = _pixel_point(landmarks[61], width, height)
    right_mouth_corner = _pixel_point(landmarks[291], width, height)
    chin = _pixel_point(landmarks[152], width, height)

    left_hollowness = np.linalg.norm(left_cheekbone - left_mouth_corner) / np.linalg.norm(left_cheekbone - chin)
    right_hollowness = np.linalg.norm(right_cheekbone - right_mouth_corner) / np.linalg.norm(right_cheekbone - chin)

    return (left_hollowness + right_hollowness) / 2.0

def _eyebrow_position(landmarks, width, height):
    """Returns a ratio of eyebrow position relative to the eyes.

    This is computed as the distance from the eyebrow to the eye, divided by
    the distance from the eye to the chin. A larger ratio indicates a higher
    eyebrow position.
    """
    left_eyebrow = _pixel_point(landmarks[105], width, height)
    right_eyebrow = _pixel_point(landmarks[334], width, height)
    left_eye = _pixel_point(landmarks[159], width, height)
    right_eye = _pixel_point(landmarks[386], width, height)
    chin = _pixel_point(landmarks[152], width, height)

    left_ratio = np.linalg.norm(left_eyebrow - left_eye) / np.linalg.norm(left_eye - chin)
    right_ratio = np.linalg.norm(right_eyebrow - right_eye) / np.linalg.norm(right_eye - chin)

    return (left_ratio + right_ratio) / 2.0

def _mouth_openness(landmarks, width, height):
    """Returns a ratio of mouth openness.

    This is computed as the distance between the upper and lower lips, divided
    by the distance from the upper lip to the chin. A larger ratio indicates a
    more open mouth.
    """
    upper_lip = _pixel_point(landmarks[13], width, height)
    lower_lip = _pixel_point(landmarks[14], width, height)
    chin = _pixel_point(landmarks[152], width, height)

    return np.linalg.norm(upper_lip - lower_lip) / np.linalg.norm(upper_lip - chin)

def _eye_openness(landmarks, width, height):
    """Returns a ratio of eye openness.

    This is computed as the distance between the upper and lower eyelids, divided
    by the distance from the upper eyelid to the chin. A larger ratio indicates a
    more open eye.
    """
    left_upper_eyelid = _pixel_point(landmarks[159], width, height)
    left_lower_eyelid = _pixel_point(landmarks[145], width, height)
    right_upper_eyelid = _pixel_point(landmarks[386], width, height)
    right_lower_eyelid = _pixel_point(landmarks[374], width, height)
    chin = _pixel_point(landmarks[152], width, height)

    left_ratio = np.linalg.norm(left_upper_eyelid - left_lower_eyelid) / np.linalg.norm(left_upper_eyelid - chin)
    right_ratio = np.linalg.norm(right_upper_eyelid - right_lower_eyelid) / np.linalg.norm(right_upper_eyelid - chin)

    return (left_ratio + right_ratio) / 2.0

def _eyelid_shape(landmarks, width, height):
    """Returns a ratio of eyelid shape.

    This is computed as the distance between the inner and outer corners of the eye,
    divided by the distance from the upper eyelid to the lower eyelid. A larger ratio
    indicates a more elongated eye shape.
    """
    left_inner_corner = _pixel_point(landmarks[133], width, height)
    left_outer_corner = _pixel_point(landmarks[33], width, height)
    right_inner_corner = _pixel_point(landmarks[362], width, height)
    right_outer_corner = _pixel_point(landmarks[263], width, height)
    left_upper_eyelid = _pixel_point(landmarks[159], width, height)
    left_lower_eyelid = _pixel_point(landmarks[145], width, height)
    right_upper_eyelid = _pixel_point(landmarks[386], width, height)
    right_lower_eyelid = _pixel_point(landmarks[374], width, height)

    left_ratio = np.linalg.norm(left_inner_corner - left_outer_corner) / np.linalg.norm(left_upper_eyelid - left_lower_eyelid)
    right_ratio = np.linalg.norm(right_inner_corner - right_outer_corner) / np.linalg.norm(right_upper_eyelid - right_lower_eyelid)

    return (left_ratio + right_ratio) / 2.0


def _mouth_width(landmarks, width, height):
    """Returns a ratio of mouth width.

    This is computed as the distance between the left and right corners of the mouth,
    divided by the distance from the upper lip to the chin. A larger ratio indicates a
    wider mouth.
    """
    left_mouth_corner = _pixel_point(landmarks[61], width, height)
    right_mouth_corner = _pixel_point(landmarks[291], width, height)
    upper_lip = _pixel_point(landmarks[13], width, height)
    chin = _pixel_point(landmarks[152], width, height)

    return np.linalg.norm(left_mouth_corner - right_mouth_corner) / np.linalg.norm(upper_lip - chin)

def _left_eyebrow_position(landmarks, width, height):
    """Returns a ratio of left eyebrow position relative to the left eye.

    This is computed as the distance from the left eyebrow to the left eye, divided by
    the distance from the left eye to the chin. A larger ratio indicates a higher
    left eyebrow position.
    """
    left_eyebrow = _pixel_point(landmarks[105], width, height)
    left_eye = _pixel_point(landmarks[159], width, height)
    chin = _pixel_point(landmarks[152], width, height)

    return np.linalg.norm(left_eyebrow - left_eye) / np.linalg.norm(left_eye - chin)

def _right_eyebrow_position(landmarks, width, height):
    """Returns a ratio of right eyebrow position relative to the right eye.

    This is computed as the distance from the right eyebrow to the right eye, divided by
    the distance from the right eye to the chin. A larger ratio indicates a higher
    right eyebrow position.
    """
    right_eyebrow = _pixel_point(landmarks[359], width, height)
    right_eye = _pixel_point(landmarks[386], width, height)
    chin = _pixel_point(landmarks[152], width, height)

    return np.linalg.norm(right_eyebrow - right_eye) / np.linalg.norm(right_eye - chin)

def _eyebrow_asymmetry(landmarks, width, height):
    """Returns a ratio of eyebrow asymmetry.

    This is computed as the absolute difference between the left and right eyebrow positions,
    divided by the distance from the left eye to the right eye. A larger ratio indicates more
    asymmetry between the eyebrows.
    """
    left_eyebrow = _pixel_point(landmarks[105], width, height)
    right_eyebrow = _pixel_point(landmarks[359], width, height)
    left_eye = _pixel_point(landmarks[159], width, height)
    right_eye = _pixel_point(landmarks[386], width, height)

    eyebrow_distance = np.linalg.norm(left_eyebrow - right_eyebrow)
    eye_distance = np.linalg.norm(left_eye - right_eye)

    return eyebrow_distance / eye_distance

def _head_pitch(landmarks, width, height):
    """Returns the head pitch angle in degrees.

    This is computed as the angle between the line connecting the chin (landmark 152)
    and the forehead (landmark 10) and the horizontal axis. A positive angle indicates
    that the head is tilted upwards, while a negative angle indicates that the head is
    tilted downwards.
    """
    chin = _pixel_point(landmarks[152], width, height)
    forehead = _pixel_point(landmarks[10], width, height)

    delta_y = forehead[1] - chin[1]
    delta_x = forehead[0] - chin[0]

    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def _head_yaw(landmarks, width, height):
    """Returns the head yaw angle in degrees.

    This is computed as the angle between the line connecting the left cheekbone (landmark 234)
    and the right cheekbone (landmark 454) and the horizontal axis. A positive angle indicates
    that the head is turned to the left, while a negative angle indicates that the head is
    turned to the right.
    """
    left_cheekbone = _pixel_point(landmarks[234], width, height)
    right_cheekbone = _pixel_point(landmarks[454], width, height)

    delta_y = right_cheekbone[1] - left_cheekbone[1]
    delta_x = right_cheekbone[0] - left_cheekbone[0]

    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def _head_roll(landmarks, width, height):
    """Returns the head roll angle in degrees.

    This is computed as the angle between the line connecting the left eye (landmark 159)
    and the right eye (landmark 386) and the horizontal axis. A positive angle indicates
    that the head is tilted to the left, while a negative angle indicates that the head is
    tilted to the right.
    """
    left_eye = _pixel_point(landmarks[159], width, height)
    right_eye = _pixel_point(landmarks[386], width, height)

    delta_y = right_eye[1] - left_eye[1]
    delta_x = right_eye[0] - left_eye[0]

    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def extract_features(landmarks, width, height) -> dict:
    """Returns a dict of {feature_name: value} for one face.

    Every feature here is normalized against face_height (landmark 10, top of
    forehead, to landmark 152, chin) — a single independent, stable reference
    measurement. This matters: dividing one raw scale-dependent measurement
    by another raw scale-dependent measurement (like eye_distance by
    cheekbone_distance) only partially cancels camera-distance effects, since
    both numbers move together as the face gets closer/farther from camera.
    Using one consistent reference for everything actually removes it.

    Add new features by following this same pattern: pick landmark indices
    (verify them against a MediaPipe face mesh index map — it's easy to grab
    the wrong point), compute a raw distance, then divide by face_height.
    """
    
    features = {
        "jaw_angle": _jaw_angle(landmarks, width, height),
        "cheek_hollowness": _cheek_hollowness(landmarks, width, height),
        "eyebrow_position": _eyebrow_position(landmarks, width, height),
        "mouth_openness": _mouth_openness(landmarks, width, height),
        "eye_openness": _eye_openness(landmarks, width, height),
        "eyelid_shape": _eyelid_shape(landmarks, width, height),
        "mouth_width": _mouth_width(landmarks, width, height),
        "left_eyebrow_position": _left_eyebrow_position(landmarks, width, height),
        "right_eyebrow_position": _right_eyebrow_position(landmarks, width, height),
        "eyebrow_asymmetry": _eyebrow_asymmetry(landmarks, width, height),
        "head_pitch": _head_pitch(landmarks, width, height),
        "head_yaw": _head_yaw(landmarks, width, height),
        "head_roll": _head_roll(landmarks, width, height),
    }

    return features