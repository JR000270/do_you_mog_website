"""
Trains a logistic regression mog-classifier on features.csv (from build_dataset.py).

USAGE:
    python train_model.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

FEATURES_CSV = "features.csv"
MODEL_OUT = "mog_model.joblib"


def main():
    df = pd.read_csv(FEATURES_CSV)
    feature_names = [c for c in df.columns if c not in ("filename", "label")]

    X = df[feature_names]
    y = df["label"]

    # At ~60 total images, a single 80/20 split leaves only ~12 test images —
    # one or two misclassifications swing accuracy by 8-16 points. 5-fold
    # cross-validation rotates through 5 different splits and averages the
    # result, giving a far more stable estimate of real performance.
    model = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv)

    print(f"Cross-validated accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
    print(f"Per-fold scores: {[round(s, 2) for s in scores]}")

    # Cross-validation is only for *evaluating* the approach — the actual
    # model you save should be trained on ALL your data, since more training
    # data generally means a better model, and you're not testing this copy.
    model.fit(X, y)

    # This is your per-feature "story" — since logistic regression is a
    # weighted sum under the hood, each coefficient tells you how much that
    # feature pushes the prediction toward mog (positive) or away (negative),
    # and the magnitude signals how strongly it swings a given photo's score.
    print("\nFeature coefficients (sorted by influence):")
    coeffs = sorted(zip(feature_names, model.coef_[0]), key=lambda x: -abs(x[1]))
    for name, coef in coeffs:
        direction = "→ mog" if coef > 0 else "→ not mog"
        print(f"  {name:25s} {coef:+.4f}  {direction}")

    joblib.dump({"model": model, "feature_names": feature_names}, MODEL_OUT)
    print(f"\nSaved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()