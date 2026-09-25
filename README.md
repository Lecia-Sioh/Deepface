DeepFace Watchlist & Face Recognition System

A Streamlit-based face recognition and watchlist system using OpenCV Haar Cascade for face detection and DeepFace with FaceNet512 for facial embeddings.

Overview

The system uses:

Streamlit — user interface and live camera feed

OpenCV Haar Cascade — detects faces

DeepFace — generates facial embeddings

FaceNet512 — the DeepFace model used for embeddings

NumPy — cosine-similarity comparison

FastAPI — receives the recognition result from Streamlit

Recognition flow

Camera
  ↓
Haar Cascade
  ↓
Face Crop
  ↓
DeepFace / FaceNet512
  ↓
Face Embedding
  ↓
Cosine Distance Matching
  ↓
Person + Entity + Confidence
  ↓
API

Features

Face Registration

Users can:

Enter a person's name

Select Authorised or Blacklisted

Upload face photos

Capture a face using the webcam

The registration process detects/crops the face, creates augmentations, generates FaceNet512 embeddings, and updates the local embedding cache.

Live Face Recognition

The live camera:

Captures frames.

Detects faces using Haar Cascade.

Crops the detected face.

Generates a FaceNet512 embedding using DeepFace.

Compares the embedding with registered embeddings.

Determines the closest match.

Calculates the similarity percentage.

Displays the result.

Sends the identification information to the API.

DeepFace

DeepFace is used through:

from deepface import DeepFace

The project calls:

DeepFace.represent(
    img_path=face_img,
    model_name="Facenet512",
    detector_backend="skip",
    enforce_detection=False
)

What each component does

Haar Cascade
→ Finds where the face is.

DeepFace + FaceNet512
→ Converts the face into a numerical embedding.

Matching code
→ Compares the embedding with registered embeddings.

Recognition Threshold

The current threshold is:

MATCH_THRESHOLD = 0.38

This is a cosine distance threshold, not a DeepFace confidence threshold.

A match is accepted when:

cosine distance <= 0.38

The displayed similarity is calculated as:

confidence_pct = (1.0 - min_dist) * 100.0

With the current threshold:

distance <= 0.38
similarity >= 62%

The displayed percentage is a cosine-similarity-based score, not a calibrated probability of recognition accuracy.

Face Augmentation

During registration, the system creates:

Original

Horizontal flip

+5° rotation

-5° rotation

Brighter

Darker

Higher contrast

These samples are converted into embeddings and used for matching.

Local Files

faces/
    └── Person Name/
        ├── sample_01.jpg
        ├── sample_02.jpg
        └── ...

watchlist.csv
embeddings_cache.pkl
haarcascade_frontalface_default.xml

faces/

Contains registered face images.

watchlist.csv

Contains registered names and watchlist status.

embeddings_cache.pkl

Contains generated facial embeddings used for matching.

haarcascade_frontalface_default.xml

OpenCV Haar Cascade model used for face detection.

API Integration

The recognition stays in Streamlit. After recognition, Streamlit sends these four pieces of information to /identify:

👤 Person name
🖼️ Person picture
🟢/🔴 Entity
📊 Confidence

Example:

{
  "person": "Lecia",
  "picture": "identified_face.jpg",
  "entity": "good",
  "confidence": 87.5
}

The API does not perform Haar Cascade detection, DeepFace embedding generation, or face matching.

API Endpoints

POST /register

Request:

name
list_type
photo

Where:

white = authorised
black = blacklisted

Response:

{
  "status": "registered"
}

POST /identify

Request:

person
picture
entity
confidence

Example response:

{
  "person": "Lecia",
  "picture": "http://127.0.0.1:8000/identify/picture",
  "entity": "good",
  "confidence": 87.5
}

GET /identify

Returns the latest identification result exposed by the API.

Example:

{
  "person": "Lecia",
  "picture": "http://127.0.0.1:8000/identify/picture",
  "entity": "good",
  "confidence": 87.5
}

GET /identify/picture

Returns the latest face picture received by the API.

Installation

Install the required packages:

pip install streamlit
pip install opencv-python
pip install numpy
pip install pandas
pip install requests
pip install deepface
pip install tensorflow
pip install fastapi
pip install uvicorn
pip install python-multipart

Running Streamlit

streamlit run streamlit.py

Replace streamlit.py with your actual Streamlit filename.

Running the API Locally

python -m uvicorn api_latest_result:app --host 127.0.0.1 --port 8000

Local API:

http://127.0.0.1:8000

FastAPI documentation:

http://127.0.0.1:8000/docs

Testing

Open:

http://127.0.0.1:8000/docs

Use the Swagger interface to test the POST endpoints.

After Streamlit sends an identification result successfully, open:

http://127.0.0.1:8000/identify

to view the latest result.

Open:

http://127.0.0.1:8000/identify/picture

to view the latest face picture.

Performance

DeepFace/FaceNet512 is the most computationally expensive part of the live recognition pipeline.

For a smoother live feed:

Use 640x480 camera resolution

Avoid running DeepFace on every frame

Recognize every few frames while displaying every frame

Reduce unnecessary API requests

Reuse Streamlit placeholders

Resize frames for Haar Cascade detection when appropriate

Example:

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

Troubleshooting

ModuleNotFoundError: deepface

pip install deepface

Camera cannot open

Check camera connection, Windows camera permissions, and whether another program is using the camera.

API connection refused

Make sure FastAPI is running:

python -m uvicorn api_latest_result:app --host 127.0.0.1 --port 8000

422 Unprocessable Content

Check that the request fields match the API.

For /identify:

person
picture
entity
confidence

405 Method Not Allowed

This usually means a browser sent GET to an endpoint that only accepts POST.

Use /docs to test POST endpoints.

Important Note

The displayed recognition score is calculated from cosine similarity:

(1 - cosine distance) × 100

It should not be interpreted as a guaranteed probability that the identity is correct.

The current matching approach compares the live embedding against the registered embeddings and selects the closest match, subject to the configured cosine-distance threshold.
