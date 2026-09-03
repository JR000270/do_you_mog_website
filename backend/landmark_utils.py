"""
Shared MediaPipe face landmark detection logic.

This is factored out of the original standalone script so the same detector
setup + extraction logic can be reused by build_dataset.py, train_model.py,
and later the FastAPI endpoint, instead of being duplicated in each.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmarker.task"


def create_detector(num_faces: int = 1) -> vision.FaceLandmarker:
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=num_faces,
    )
    return vision.FaceLandmarker.create_from_options(options)


def get_landmarks(image_path: str, detector: vision.FaceLandmarker):
    """Returns (landmarks, width, height) for the first detected face,
    or None if the image couldn't be read or no face was found.

    Reusing the same detector instance across many calls (rather than
    creating a new one per image) matters once you're processing a whole
    dataset folder — model setup has real overhead you don't want to pay
    per-image.
    """
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        return None

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    detection_result = detector.detect(image)
    if not detection_result.face_landmarks:
        return None

    landmarks = detection_result.face_landmarks[0]
    height, width = rgb_image.shape[:2]
    return landmarks, width, height
