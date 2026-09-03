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

    #get the components of the prediction
    coefficients = model.coef_[0]

    #dictionary of contributions of each feature to the prediction. feature: contribution pairs,
    #where contribution = model coefficient * feature_value
    contributions = {
        feature_names[i]: coefficients[i] * x[feature_names[i]].iloc[0]
        for i, feature in enumerate(feature_names)
    }

    results["mog_probability"] = mog_probability
    results["contributions"] = contributions

    return results


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

