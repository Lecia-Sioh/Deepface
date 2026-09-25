# DeepFace Watchlist & Recognition System

A face recognition watchlist system built with **DeepFace**, **FaceNet512**, **OpenCV**, and **Streamlit**.

The system is designed to register faces, generate facial embeddings, compare new face images against the registered database, and classify recognised people as **Authorised** or **Blacklisted**.

---

## Project Overview

The system uses DeepFace to generate facial embeddings with the **FaceNet512** model.

The overall recognition flow is:

```text
Input Face Image
      ↓
Haar Cascade
      ↓
Face Detection & Cropping
      ↓
DeepFace + FaceNet512
      ↓
Face Embedding
      ↓
Cosine Distance Matching
      ↓
Person Identification
      ↓
Entity + Confidence
      ↓
API
```

### Main Technologies

| Technology | Purpose |
|---|---|
| **DeepFace** | Facial representation and embedding generation |
| **FaceNet512** | Model used by DeepFace |
| **OpenCV** | Face detection, image processing, and face cropping |
| **NumPy** | Embedding comparison and cosine similarity |
| **Pandas** | Watchlist management |
| **Streamlit** | Application interface |
| **FastAPI** | API communication |

---

## How DeepFace Is Used

DeepFace is imported with:

```python
from deepface import DeepFace
```

The project uses:

```python
DeepFace.represent(
    img_path=face_img,
    model_name="Facenet512",
    detector_backend="skip",
    enforce_detection=False
)
```

DeepFace converts a detected face into a numerical **FaceNet512 embedding**.

The embedding is then used by the matching system to find the closest registered person.

### DeepFace vs Haar Cascade

These two components have different roles:

**Haar Cascade**

> Detects where the face is located in the image.

**DeepFace + FaceNet512**

> Converts the detected face into a numerical representation that can be compared with registered faces.

---

## Face Registration

When a person is registered, the system:

1. Takes the person's name.
2. Assigns a watchlist status.
3. Receives one or more face images.
4. Detects and crops the face.
5. Generates augmented versions of the image.
6. Generates FaceNet512 embeddings using DeepFace.
7. Saves the embeddings for future matching.

### Watchlist Status

A person can be registered as:

```text
Authorised
Blacklisted
```

These statuses are later converted into API entities:

```text
Authorised  →  good
Blacklisted →  bad
```

---

## Face Augmentation

The system creates additional versions of a registered face:

- Original
- Horizontal flip
- +5° rotation
- -5° rotation
- Brighter
- Darker
- Higher contrast

Each augmented image can produce an additional FaceNet512 embedding.

This gives the matcher multiple representations of the same person.

---

## Face Matching

The system compares the new face embedding with the registered embeddings using cosine similarity / cosine distance.

The current threshold is:

```python
MATCH_THRESHOLD = 0.8
```

A match is accepted when:

```text
Cosine Distance ≤ 0.8
```

The displayed similarity is calculated as:

```python
confidence_pct = (1.0 - min_dist) * 100.0
```

For example:

```text
Distance:   0.13
Similarity: 87%
```

The displayed percentage is a **similarity score**, not a guaranteed probability of recognition accuracy.

---

## Streamlit Application

The Streamlit application provides three main functions:

### 1. Register Face

Register a new person and generate the embeddings used by the recognition system.

### 2. De-register Face

Remove a registered person from the local watchlist and rebuild the embedding cache.

### 3. Live Recognition

Process a face, generate its embedding, compare it with the registered embeddings, and display:

```text
Person
Status
Confidence
```

The recognition logic remains inside the Streamlit application.

---

## API Integration

After recognition, the result can be sent to the API.

The `/identify` information contains:

```text
👤 Person name
🖼️ Person picture
🟢/🔴 Entity
📊 Confidence
```

Example:

```json
{
    "person": "Lecia",
    "entity": "good",
    "confidence": 87.5
}
```

The API is used to receive and provide the recognition information. The face recognition itself is performed by the Streamlit application using DeepFace.

### Entity Values

```text
good    → Authorised person
bad     → Blacklisted person
unknown → No accepted match
```

---

## API Endpoints

### `POST /register`

Used to send registration information.

Request:

```text
name
list_type
photo
```

Example:

```json
{
    "name": "Lecia",
    "list_type": "white"
}
```

Response:

```json
{
    "status": "registered"
}
```

### `POST /identify`

Used to send the identification result.

Request:

```text
person
picture
entity
confidence
```

Example response:

```json
{
    "person": "Lecia",
    "picture": "http://127.0.0.1:8000/identify/picture",
    "entity": "good",
    "confidence": 87.5
}
```

### `GET /identify`

Returns the latest identification information exposed by the API.

Example:

```json
{
    "person": "Lecia",
    "picture": "http://127.0.0.1:8000/identify/picture",
    "entity": "good",
    "confidence": 87.5
}
```

---

## Project Files

The main local data used by the application includes:

```text
faces/
watchlist.csv
embeddings_cache.pkl
haarcascade_frontalface_default.xml
```

### `faces/`

Contains the registered face images.

### `watchlist.csv`

Contains the registered person's name and watchlist status.

### `embeddings_cache.pkl`

Contains the generated face embeddings used for matching.

### `haarcascade_frontalface_default.xml`

Used for face detection before the face is passed to DeepFace.

---

## Installation

Install the main dependencies:

```bash
pip install deepface
pip install tensorflow
pip install opencv-python
pip install numpy
pip install pandas
pip install streamlit
pip install requests
pip install fastapi
pip install uvicorn
pip install python-multipart
```

---

## Running the Application

### Streamlit

Run the Streamlit application with:

```bash
streamlit run streamlit.py
```

Replace `streamlit.py` with the filename of your application.

### FastAPI

Run the local API with:

```bash
python -m uvicorn api_latest_result:app --host 127.0.0.1 --port 8000
```

API address:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Example Recognition Result

A successful match may produce:

```text
Person: Lecia
Entity: good
Confidence: 87.5%
```

The system then sends the recognition information to the API.

---

## Key Point

The core of this project is:

```text
DeepFace
     +
FaceNet512
     +
Face Embeddings
     +
Cosine Similarity
     +
Watchlist Matching
     +
API Integration
```

DeepFace is responsible for creating the facial representation used by the recognition system, while the application's matching logic determines which registered person is the closest match.
