"""
Tests the trained mog-classifier against the labeled images in
data/test/mog and data/test/not_mog, and reports overall accuracy.
command:
    python test_model.py
"""

from pathlib import Path

import joblib
import pandas as pd

from landmark_utils import create_detector, get_landmarks
from features import extract_features

MODEL_PATH = "mog_model.joblib"
TEST_DIR = Path("../data/test")
LABELS = {"mog": 1, "not_mog": 0}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_names = saved["feature_names"]

    detector = create_detector()

    correct = 0
    total = 0

    for folder_name, true_label in LABELS.items():
        folder = TEST_DIR / folder_name
        if not folder.exists():
            print(f"Warning: {folder} does not exist, skipping.")
            continue

        image_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

        for image_path in image_paths:
            result = get_landmarks(str(image_path), detector)
            if result is None:
                print(f"{image_path.name:20s}  no face detected — skipped")
                continue

            landmarks, width, height = result
            features = extract_features(landmarks, width, height)

            # A one-row DataFrame, columns named and ordered exactly like the
            # training data (train_model.py fit on df[feature_names]). Passing a
            # plain list would work numerically, but sklearn couldn't verify the
            # columns line up and would warn about it.
            x = pd.DataFrame([features], columns=feature_names)

            predicted_label = model.predict(x)[0]
            mog_probability = model.predict_proba(x)[0][1]

            is_correct = predicted_label == true_label
            correct += is_correct
            total += 1

            true_str = "mog" if true_label == 1 else "not mog"
            pred_str = "mog" if predicted_label == 1 else "not mog"
            mark = "correct" if is_correct else "WRONG"
            print(
                f"{image_path.name:20s}  true={true_str:8s}  pred={pred_str:8s}  "
                f"(mog probability: {mog_probability:.2%})  {mark}"
            )

    if total == 0:
        print("\nNo labeled test images found — check data/test/mog and data/test/not_mog.")
        return

    accuracy = correct / total
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.2%})")


if __name__ == "__main__":
    main()
