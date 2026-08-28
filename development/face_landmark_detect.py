"""
Phase 1: Face landmark detection with MediaPipe Tasks.

Adapted from Google's official Face Landmarker Colab demo to run as a
standalone local script (no Colab-specific dependencies).

ONE-TIME SETUP (run in your terminal):
    pip install mediapipe opencv-python matplotlib
    curl -o face_landmarker.task -L https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

USAGE:
    python face_landmark_detect.py path/to/your/photo.jpg
"""

import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draws the face mesh, contours, and iris tracking onto a copy of the image."""
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for face_landmarks in face_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image


def main(image_path: str):
    # STEP 1: Set up the detector.
    # base_options points at the model file you downloaded above — this .task
    # file contains the actual trained neural network weights for finding faces
    # and their landmarks. FaceLandmarkerOptions configures what we want back:
    #   - output_face_blendshapes: expressiveness scores (smile, brow raise, etc.)
    #     we won't need these for the mog-scorer, but they're free to grab
    #   - num_faces: cap how many faces to detect (1 is right for your use case —
    #     a single selfie/upload)
    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # STEP 2: Load the image.
    # We load through OpenCV rather than mp.Image.create_from_file, because
    # cv2.imread always normalizes to 3-channel BGR — it silently drops any
    # alpha (transparency) channel a PNG might have. mp.Image.create_from_file
    # instead preserves whatever channel count the source file has, which
    # breaks MediaPipe's drawing step on 4-channel (RGBA) PNGs. Since you can't
    # control what file formats users upload later, always going through
    # OpenCV first keeps this predictable.
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        print(f"Could not read image at {image_path}")
        return
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # mp.Image wraps a plain numpy array in MediaPipe's internal format.
    # image_format=SRGB tells it the data is standard 3-channel color.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # STEP 3: Run detection.
    # This is the actual inference call — the model looks at the image and
    # returns landmark coordinates.
    detection_result = detector.detect(image)

    if not detection_result.face_landmarks:
        print("No face detected in this image.")
        return

    # STEP 4: Inspect the raw output.
    # face_landmarks is a list (one entry per detected face — we asked for
    # num_faces=1, so there's just one). Each entry is a list of 468 points,
    # and each point has .x, .y, .z — x and y are NORMALIZED, meaning they're
    # fractions between 0 and 1 relative to image width/height, not pixel
    # coordinates. You'll need to multiply by image width/height to get pixels
    # when you start measuring distances in Phase 2.
    landmarks = detection_result.face_landmarks[0]
    print(f"Detected {len(landmarks)} landmarks.")
    print("First 5 landmarks (normalized x, y, z):")
    for i, lm in enumerate(landmarks[:5]):
        print(f"  [{i}] x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f}")

    #messing with landmarks section
    eye_distance = eye_distance_calculation(landmarks, rgb_image.shape[1], rgb_image.shape[0])
    cheekbones_distance = cheekbones_distance_calculation(landmarks, rgb_image.shape[1], rgb_image.shape[0])
    #print(f"Distance between eyes: {eye_distance:.2f} pixels")


    # STEP 5: Visualize the result.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)


    # Save to disk (always works, headless-safe) and also try to pop up a
    # window via matplotlib so you can eyeball it immediately.
    output_path = "annotated_" + image_path.split("/")[-1]
    cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print(f"Saved annotated image to {output_path}")
    print(f"Distance between eyes: {eye_distance:.2f} pixels")
    print(f"Cheekbone prominence: {cheekbones_distance:.2f} pixels")

    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title("Face Mesh Landmarks")
    plt.suptitle(f"Distance between eyes: {eye_distance:.2f} pixels")
    # plt.suptitle(f"Cheekbone prominence: {cheekbones_distance:.2f} pixels", y=0.92)
    plt.show()

    


def eye_distance_calculation(landmarks, in_image_width, in_image_height):
    """Example of how to manipulate the landmark coordinates.

    This is where you could implement your own scoring logic for the mog-scorer.
    For example, you could measure distances between landmarks, compute angles,
    or apply any other geometric analysis to determine facial expressiveness.

    Args:
        landmarks: A list of normalized landmark points (x, y, z).
        in_image_width: The width of the input image.
        in_image_height: The height of the input image.
    """

    # Example: Print the distance between the left and right eye outer or inner corners
    left_eye_iner = landmarks[133]  # Left eye inner corner
    right_eye_inner = landmarks[362]  # Right eye inner corner
    left_eye_outer = landmarks[33]  # Left eye outer corner
    right_eye_outer = landmarks[263]  # Right eye outer corner

    # Convert normalized coordinates to pixel coordinates
    image_width = in_image_width  # Replace with actual image width
    image_height = in_image_height  # Replace with actual image height
    left_eye_pixel = (int(left_eye_outer.x * image_width), int(left_eye_outer.y * image_height))
    right_eye_pixel = (int(right_eye_outer.x * image_width), int(right_eye_outer.y * image_height))

    eye_distance = np.sqrt((right_eye_pixel[0] - left_eye_pixel[0]) ** 2 +
                       (right_eye_pixel[1] - left_eye_pixel[1]) ** 2)
    
    return eye_distance


def cheekbones_distance_calculation(landmarks, in_image_width, in_image_height):
    """Example of how to manipulate the landmark coordinates.

    This is where you could implement your own scoring logic for the mog-scorer.
    For example, you could measure distances between landmarks, compute angles,
    or apply any other geometric analysis to determine facial expressiveness.

    Args:
        landmarks: A list of normalized landmark points (x, y, z).
        in_image_width: The width of the input image.
        in_image_height: The height of the input image.
    """

    # Example: Print the distance between the left and right cheekbones
    left_cheekbone = landmarks[234]  # Left cheekbone
    right_cheekbone = landmarks[454]  # Right cheekbone

    # Convert normalized coordinates to pixel coordinates
    image_width = in_image_width  # Replace with actual image width
    image_height = in_image_height  # Replace with actual image height
    left_cheekbone_pixel = (int(left_cheekbone.x * image_width), int(left_cheekbone.y * image_height))
    right_cheekbone_pixel = (int(right_cheekbone.x * image_width), int(right_cheekbone.y * image_height))

    cheekbones_distance = np.sqrt((right_cheekbone_pixel[0] - left_cheekbone_pixel[0]) ** 2 +
                       (right_cheekbone_pixel[1] - left_cheekbone_pixel[1]) ** 2)
    
    return cheekbones_distance

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python face_landmark_detect.py path/to/your/photo.jpg")
        sys.exit(1)
    main(sys.argv[1])