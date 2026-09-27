"""
Trains a logistic regression mog-classifier on features.csv (from build_dataset.py).
uses scikit learn on the csv of features to train a logistic regression model, and saves it to mog_model.joblib.
command:
    python train_model.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_CSV = "features.csv"
MODEL_OUT = "mog_model.joblib"


def main():
    #turn the raw csv into a pandas dataframe, split it into features and labels
    df = pd.read_csv(FEATURES_CSV)
    feature_names = [c for c in df.columns if c not in ("filename", "label")]

    X = df[feature_names]
    y = df["label"]

    #  5-fold cross-validation rotates through 5 different splits and averages the
    # result, giving a far more stable estimate of real performance.
    # The features mix wildly different natural scales — ratios like
    # "hollow cheeks" (~0-2) next to angles like "head pitch" (~-90 to 90).
    # Without StandardScaler, LogisticRegression still fits fine, but the
    # resulting coefficients aren't comparable to each other: a feature's
    # coefficient partly just reflects "how big are this feature's raw
    # values", not how predictive it is. Scaling puts every feature in units
    # of standard deviations from its mean, so coefficients (and later,
    # coef * scaled_value contributions) are actually comparable across
    # features.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv)

    print(f"Cross-validated accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
    print(f"Per-fold scores: {[round(s, 2) for s in scores]}")

    # Cross-validation is only for *evaluating* the approach — the actual
    # model you save should be trained on ALL your data, since more training
    # data generally means a better model, and you're not testing this copy.
    model.fit(X, y)

    # This is your per-feature scaling since logistic regression is a
    # weighted sum under the hood, each coefficient tells you how much that
    # feature pushes the prediction toward mog (positive) or away (negative),
    # and the magnitude signals how strongly it swings a given photo's score.
    # These coefficients are on standardized features, so magnitudes are now
    # directly comparable across features.
    clf = model.named_steps["clf"]
    print("\nFeature coefficients (sorted by influence, on standardized features):")
    coeffs = sorted(zip(feature_names, clf.coef_[0]), key=lambda x: -abs(x[1]))
    for name, coef in coeffs:
        direction = "-> mog" if coef > 0 else "-> not mog"
        print(f"  {name:25s} {coef:+.4f}  {direction}")

    # Per-feature calibration for the frontend's 1-10 "feature rating" display.
    # The old approach min-max scaled a photo's top-5 contributions *against
    # each other*, so whichever feature naturally swings hardest (bigger
    # coefficient, wider spread, whatever) always hit 10 and forced the other
    # four toward 1 - regardless of whether those four were actually weak.
    # Instead, score each feature against the range of contributions that particular
    # feature produces across the training set, independent of what else
    # shows up in a given photo's top 5.
    #
    # Using the 5th/95th percentile rather than raw min/max keeps one unusual
    # training photo from single-handedly setting a feature's whole scale -
    # with ~119 rows that excludes ~6 rows on each tail.
    scaler_fitted = model.named_steps["scaler"]
    x_scaled_all = scaler_fitted.transform(X)
    contributions_all = x_scaled_all * clf.coef_[0]  # broadcasts per column
    feature_bounds = {
        name: (
            float(np.percentile(contributions_all[:, i], 5)),
            float(np.percentile(contributions_all[:, i], 95)),
        )
        for i, name in enumerate(feature_names)
    }

    joblib.dump(
        {"model": model, "feature_names": feature_names, "feature_bounds": feature_bounds},
        MODEL_OUT,
    )
    print(f"\nSaved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()