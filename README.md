# Do You Mog?

A web app that scores how much a face "mogs" (a slang term for having a strong, dominant facial appearance). Upload a photo and a logistic regression model — trained on facial landmark geometry — returns a mog probability along with a breakdown of which facial features pushed the score up or down.

## How it works

1. The React frontend sends an uploaded image to a FastAPI backend.
2. The backend runs [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) on the image to extract 3D facial landmarks.
3. Landmark coordinates are converted into named geometric features — jaw angle, cheek hollowness, eye/mouth openness, eyebrow position and asymmetry, head pose, etc. (see [backend/features.py](backend/features.py)).
4. A pre-trained scikit-learn `LogisticRegression` model (`mog_model.joblib`) scores the feature vector and returns a mog probability, plus each feature's contribution to that score.
5. The frontend displays the probability and a per-feature breakdown.

## Tech stack

**Frontend:** React 19, Vite, Tailwind CSS, Axios
**Backend:** FastAPI, Uvicorn, MediaPipe, OpenCV, scikit-learn, pandas, joblib

## Project structure

```
backend/          FastAPI app served in production — the /analyze endpoint,
                   feature extraction, landmark detection, and the trained model
frontend/          React + Vite single-page app
data/              Labeled training/test images (mog vs. not_mog)
development/       Scripts used to build the dataset and train the model
                    (collect_images.py, build_dataset.py, train_model.py, ...)
```

## Running locally

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The API runs at `http://localhost:8000`. `POST /analyze` accepts a multipart file upload (`file`) and returns `{ mog_probability, contributions }`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The app runs at `http://localhost:5173` and calls the backend at `http://localhost:8000`.

## Retraining the model

The `development/` folder holds the scripts used to build `mog_model.joblib`:

1. `collect_images.py` — gather labeled face images
2. `build_dataset.py` — run landmark detection + feature extraction over `data/`, producing `features.csv`
3. `train_model.py` — cross-validate and fit a `LogisticRegression` on `features.csv`, saving the result to `mog_model.joblib`

Copy the resulting `mog_model.joblib` into `backend/` to use it with the API.
