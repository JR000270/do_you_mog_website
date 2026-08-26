"""
Walks labeled image folders and extracts features into a CSV for training.

Expected folder layout (create this alongside your scripts):
    data/
        mog/          <- photos that ARE making the mog face
            photo1.jpg
            photo2.png
            ...
        not_mog/      <- photos that are NOT
            photo1.jpg
            ...

USAGE:
    python build_dataset.py
"""

import csv
from pathlib import Path

from landmark_utils import create_detector, get_landmarks
from features import extract_features

DATA_DIR = Path("data")
LABELS = {"mog": 1, "not_mog": 0}
OUTPUT_CSV = "features.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    detector = create_detector()
    rows = []
    feature_names = None

    for folder_name, label in LABELS.items():
        folder = DATA_DIR / folder_name
        if not folder.exists():
            print(f"Warning: {folder} does not exist, skipping.")
            continue

        image_paths = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        print(f"Processing {len(image_paths)} images in {folder}...")

        for image_path in image_paths:
            result = get_landmarks(str(image_path), detector)
            if result is None:
                print(f"  Skipping {image_path.name} — no face detected.")
                continue

            landmarks, width, height = result
            features = extract_features(landmarks, width, height)

            if feature_names is None:
                feature_names = list(features.keys())

            row = {"filename": image_path.name, "label": label, **features}
            rows.append(row)

    if not rows:
        print("No images processed — check your data/ folder layout.")
        return

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"] + feature_names)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
