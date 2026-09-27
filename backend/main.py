from fastapi import FastAPI, UploadFile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import tempfile
import os
import random

app = FastAPI()

origins = ["http://localhost:8000", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#comment pools keyed by the lower bound of their tier (threshold <= mog_probability)
FUNNY_COMMENTS = {
    0: [
        "You are not locked in",
        "YIKES! Might wanna look up a guide",
    ],
    20: [
        "Lowkey built like a fetus ngl...",
        "Bro said 'ship it' before finishing the character creator",
        "This is giving unfinished sim character",
        "This is trash",
        "The mirror really said 'no thank you'",
        "Buddy this is a rebuild-from-scratch situation",
        "What are ya dooiin",
        "GYYYYAAAAAHHHH!!!!! LOOK AWAYYYY",
    ],
    50: [
        "You gotta lock in harder than that",
        "So close to mid, yet so far",
        "Gym membership is calling your name",
        "C++ for effort, we've seen better",
        "Almost had it, almost",
        "Mid but climbing, respect the grind",
    ],
    60: [
        "This is indeed some mogging right here",
        "Okay okay, I see you twin!",
        "Certified chad moment",
        "Respectable numbers, keep it up",
        "You're giving 'good bone structure' energy",
        "Straight up solid",
    ],
    80: [
        "Zayum! You making the camera blush!",
        "The mogging levels are booming right now!",
        "The confidence is radiating off this one",
        "You ate that, and left no crumbs",
        "Dont melt the camera hot stuff!",
        "Camera said 'thank you for your service'",
    ],
    90: [
        "SHEEESH! We gotta get you in a kitchen because you cooked!",
        "Bro is not on the same difficulty setting as the rest of us",
        "This is straight up unfair to the competition",
        "Okay Greek statue, calm down",
        "The mog is unmatched, take the W",
        "Gonna make me act up, no cap",
    ],
    95: [
        "HOLY COW BRUH! You make handsome squidward look ugly!!",
        "Bro broke the mog-o-meter",
        "This ain't even fair anymore, call the IRS because you took the whole budget",
        "Genetics really said 'let's give this one everything'",
        "Sistine Chapel ceiling but it's a face",
        "Simulation error: too much mogging loaded at once",
    ],
}

def get_funny_comment(mog_probability):
    #each pool is keyed by the *lower* bound of its tier - use the pool for the
    #highest threshold the score still meets or exceeds (e.g. 25 -> the 20 pool,
    #15 -> the 0 pool)
    rating_pool = max(threshold for threshold in FUNNY_COMMENTS if threshold <= mog_probability)
    return random.choice(FUNNY_COMMENTS[rating_pool])


def get_top_contributions(contributions, feature_bounds):
    #sorts the dictionary and returns the top 5 features with the highest positive contributions to the mogging score
    #top_sorted_contributions = dict(sorted(((key, value) for key, value in contributions.items() if value > 0), key=lambda item: item[1], reverse=True)[:5])
    #top 5 features
    top_sorted_contributions = dict(sorted(((key, value) for key, value in contributions.items()), key=lambda item: item[1], reverse=True)[:5])


    if not top_sorted_contributions:
        return {}

    #scale each feature's contribution against ITS OWN calibrated range (the
    #5th/95th percentile of that feature's contribution across the training
    #set - see train_model.py) instead of min-maxing the top 5 against each
    #other. Row-relative scaling forced every non-dominant feature toward 1
    #whenever one feature naturally swings harder than the rest; a fixed
    #per-feature scale means a rating reflects how strong that feature really
    #is, not just how it stacks up against whoever else made this top 5.
    result = {}
    for key, value in top_sorted_contributions.items():
        low, high = feature_bounds[key]
        if high == low:
            result[key] = 10
            continue
        scaled = 1 + (value - low) / (high - low) * 9
        #a live photo can fall outside the training set's calibration range,
        #so clip to keep the displayed rating within 1-10
        result[key] = round(min(10, max(1, scaled)))

    return result

#run the model on a single image that is sent to the backend from the frontend. return the result to the frontend
@app.post("/analyze", response_model=dict)
async def analyze_image(file: UploadFile) -> dict:
    #load the model
    saved = joblib.load("mog_model.joblib")
    model = saved["model"]
    feature_names = saved["feature_names"]
    feature_bounds = saved["feature_bounds"]

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
        #no face found in the image - respond with the same shape as a
        #successful analysis so the frontend can render it unchanged, just
        #with a zero score and a prompt to upload a photo with a face
        results["mog_probability"] = 0
        results["contributions"] = {}
        results["funny_comment"] = "No face detected - please upload a photo with a face!"
        return results

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
        #feature_names[i]: x_scaled[i]
        
        for i, feature in enumerate(feature_names)
    }
    best_contributions = get_top_contributions(contributions, feature_bounds) #get the top 5 features with the highest positive contributions to the mogging score

    #add the analysis results to the dictionary
    results["mog_probability"] = mog_probability
    #results["contributions"] = contributions
    results["contributions"] = best_contributions
    results["funny_comment"] = get_funny_comment(mog_probability)
    return results


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)