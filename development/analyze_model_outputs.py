"""
This script is to test the functionality of the model and to anlayze the models prediction output
 by looking at how much the features impacted the tested image. 
Using this as baseline to then setup giving some of the highest and lowest scored features to make
funny categories describing the image. The parts behind the overall "mogging score"

command(s): 
cd development
python analyze_model_outputs.py
"""

def predict_image(image):
    #load the model
    import joblib
    import pandas as pd
    saved = joblib.load("mog_model.joblib")
    model = saved["model"]
    feature_names = saved["feature_names"]

    #get the features from the image
    from features import extract_features
    from landmark_utils import create_detector, get_landmarks
    detector = create_detector()
    landmarks, width, height = get_landmarks(image, detector)
    features = extract_features(landmarks, width, height)

    # A one-row DataFrame, columns named/ordered exactly like training data —
    # sklearn needs a 2D array-like, and this also guards against a silent
    # column-order mismatch (see test_model.py for the same pattern).
    x = pd.DataFrame([features], columns=feature_names)

    #make a prediction, return the probability of being a mog
    #probability will be used as the mogging score in the frontend
    mog_probability = model.predict_proba(x)[0][1]

    #get the components of the prediction
    coefficients = model.coef_[0]

    # Logistic regression scores a face by summing coef * feature_value for
    # every feature (plus an intercept), then squashing that sum into a
    # probability. coef alone is how much a feature matters *in general* —
    # to know how much it swung THIS face's score, you need coef * this
    # face's value for that feature. That's the number worth ranking to get
    # "most mogging features" for one photo.
    contributions = {
        feature: coefficients[i] * x[feature].iloc[0]
        for i, feature in enumerate(feature_names)
    }

    #print out the contributions in a nice way, strongest impact first
    print("Feature contributions (sorted by impact on this image):")
    for feature, contribution in sorted(contributions.items(), key=lambda item: -abs(item[1])):
        direction = "-> mog" if contribution > 0 else "-> not mog"
        print(f"  {feature:25s} {contribution:+.4f}  {direction}")

    return mog_probability, contributions

def main():
    # Example usage of predict_image function
    test_image_path = "../data/test/mog/1.jpg"
    prediction, contributions = predict_image(test_image_path)
    print(f"\nPrediction for {test_image_path}: {prediction:.2%} mog")


if __name__ == "__main__":
    main()