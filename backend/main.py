from fastapi import FastAPI, UploadFile, HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import tempfile
import os

app = FastAPI()

origins = ["http://localhost:8000", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_funny_comment(mog_probability):
    #returns a funny comment based on the mogging score
    if mog_probability < 20:
        return "Lowkey built like a fetus ngl..."
    elif mog_probability < 50:
        return "You gotta lock in harder than that"
    elif mog_probability < 60:
        return "This is indeed some mogging right here"
    elif mog_probability < 80:
        return "Zayum! You making the camera blush!"
    elif mog_probability < 90:
        return "SHEEESH! We gotta get you in a kitchen because you cooked!"
    else:
        return "HOLY COW BRUH! You make handsome squidward look ugly!!"


def get_top_contributions(contributions):
    #sorts the dictionary and returns the top 5 features with the highest positive contributions to the mogging score
    #top_sorted_contributions = dict(sorted(((key, value) for key, value in contributions.items() if value > 0), key=lambda item: item[1], reverse=True)[:5])
    top_sorted_contributions = dict(sorted(((key, value) for key, value in contributions.items() if value > 0), key=lambda item: item[1], reverse=True))


    if not top_sorted_contributions:
        return {}

    #min-max scale the top 5 contributions against each other (not against a
    #single global max) so the spread actually uses the 1-10 range instead of
    #every non-winning feature flooring to the same value
    min_contribution = min(top_sorted_contributions.values())
    max_contribution = max(top_sorted_contributions.values())

    if max_contribution == min_contribution:
        return {key: 10 for key in top_sorted_contributions}

    return {
        key: round(1 + (value - min_contribution) / (max_contribution - min_contribution) * 9)
        for key, value in top_sorted_contributions.items()
    }

#run the model on a single image that is sent to the backend from the frontend. return the result to the frontend
@app.post("/analyze", response_model=dict)
async def analyze_image(file: UploadFile) -> dict:
    #load the model
    saved = joblib.load("mog_model.joblib")
    model = saved["model"]
    feature_names = saved["feature_names"]

    #dictionary to hold all the results
    results = {}

    #get the features from the image to give to the model
    from features import extract_features
    from landmark_utils import create_detector, get_landmarks

    #get_landmarks expects a file path, so the uploaded bytes are written to a
    #temp file on disk before being handed off
    suffix = os.path.splitext(file.filename)[1]
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        #create the detector from mediapipe and get the landmarks from the image
        detector = create_detector()
        detection = get_landmarks(tmp_path, detector) #pass image to the detector
    finally:
        os.remove(tmp_path)

    if detection is None:
        raise HTTPException(status_code=422, detail="No face detected in the uploaded image")

    landmarks, width, height = detection
    features = extract_features(landmarks, width, height) #dictionary of feature:value pairs extracted from the image

    #one-row DataFrame using the features extracted from image
    x = pd.DataFrame([features], columns=feature_names)

    #make a prediction, return the probability of being a mog
    #probability will be used as the mogging score in the frontend
    mog_probability = model.predict_proba(x)[0][1]
    mog_probability = round(mog_probability, 2) * 100 #round to 2 decimal places for frontend display


    #get the components of the prediction. the model is a Pipeline(scaler, clf) -
    #contributions must be computed on the *scaled* feature values the coefficients
    #were actually fit on, otherwise features with naturally large raw units (e.g.
    #head_pitch in degrees, ~-90) swamp features with naturally small raw units
    #(e.g. cheek_hollowness, a ratio ~0-2) regardless of how predictive each is
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    coefficients = clf.coef_[0]
    x_scaled = scaler.transform(x)[0]

    #dictionary of contributions of each feature to the prediction. feature: contribution pairs,
    #where contribution = model coefficient * standardized feature_value
    contributions = {
        feature_names[i]: coefficients[i] * x_scaled[i]
        for i, feature in enumerate(feature_names)
    }
    best_contributions = get_top_contributions(contributions) #get the top 5 features with the highest positive contributions to the mogging score

    #add the analysis results to the dictionary
    results["mog_probability"] = mog_probability
    #results["contributions"] = contributions
    results["contributions"] = best_contributions
    results["funny_comment"] = get_funny_comment(mog_probability)
    return results


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)