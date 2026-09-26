# DeepFace Watchlist & Recognition System

A face recognition watchlist system built with **DeepFace ArcFace**, **RetinaFace**, **OpenCV**, and **Streamlit**.

The system is designed to register faces, generate facial embeddings, compare new face images against the registered database, and classify recognised people as **Authorised** or **Blacklisted**.

---

## Project Overview

The system uses DeepFace to generate facial embeddings with the **ArcFace** model. It compares normalized embeddings using **Euclidean L2 distance**.

The overall recognition flow is:

```text
Input Face Image
      ↓
Haar Cascade (live frame localization)
      ↓
Face Detection & Cropping
      ↓
DeepFace ArcFace + RetinaFace
      ↓
Face Embedding
      ↓
Euclidean L2 Distance Matching
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
| **ArcFace** | Model used by DeepFace to generate face embeddings |
| **RetinaFace** | Detector used by DeepFace during embedding extraction |
| **OpenCV** | Live frame face localization, image processing, and cropping |
| **NumPy** | Embedding comparison using Euclidean L2 distance |
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
      model_name="ArcFace",
      detector_backend="retinaface",
    enforce_detection=False
)
```

DeepFace converts a face image into a numerical **ArcFace embedding**. The live camera first localizes candidate faces with OpenCV's Haar Cascade, then the cropped image is passed to DeepFace with RetinaFace configured as its detector.

The embedding is then used by the matching system to find the closest registered person.

### DeepFace vs Haar Cascade

These two components have different roles:

**Haar Cascade**

> Detects where the face is located in the image.

**DeepFace ArcFace + RetinaFace**

> Refines face detection and converts the face into a numerical representation that can be compared with registered faces.

---

## Face Registration

When a person is registered, the system:

1. Takes the person's name.
2. Assigns a watchlist status.
3. Receives one or more face images.
4. Detects and crops the face.
5. Generates augmented versions of the image.
6. Generates ArcFace embeddings using DeepFace and RetinaFace.
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

Each augmented image can produce an additional ArcFace embedding.

This gives the matcher multiple representations of the same person.

---

## Face Matching

The system compares the new face embedding with the registered embeddings using Euclidean L2 distance.

The current threshold is:

```python
MATCH_THRESHOLD = 1.13
```

A match is accepted when:

```text
Euclidean L2 Distance ≤ 1.13
```

The displayed similarity is calculated as:

```python
similarity_pct = (1.0 - min_dist / 2.0) * 100.0
```

For example:

```text
Distance:   0.26
Similarity: 87%
```

The displayed percentage is a **scaled similarity score**, not a calibrated probability of recognition accuracy. The threshold should be validated for the deployment camera and data.

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

After recognition, the result can be sent to the API. The API runs separately from the Streamlit app.

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

The face recognition itself is performed by the Streamlit application using DeepFace. The API's `/register` endpoint only reads the uploaded image and stores the latest registration metadata in process memory; it does not add a person to the Streamlit watchlist or persist the image. Register through the Streamlit application to create local watchlist entries and embeddings. API memory is cleared when the API process restarts.

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
arcface_embeddings_cache.pkl
haarcascade_frontalface_default.xml
```

### `faces/`

Contains the registered face images.

### `watchlist.csv`

Contains the registered person's name and watchlist status.

### `arcface_embeddings_cache.pkl`

Contains the generated face embeddings used for matching.

### `haarcascade_frontalface_default.xml`

Used for face detection before the face is passed to DeepFace.

---

## Installation

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

---

## Running the Application

### Streamlit

Start the Streamlit application with:

```bash
streamlit run face.py
```

### FastAPI

In a separate terminal, start the API with:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

API address:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

### Unit tests

Open `face_matching_unit_tests.ipynb` in Jupyter from the project root and run its cells in order. The tests extract and exercise the real `match_face_embedding` function using synthetic vectors; they do not process photographs, call the API, or validate the ArcFace model or camera pipeline.

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
ArcFace + RetinaFace
     +
Face Embeddings
     +
Euclidean L2 Distance
     +
Watchlist Matching
     +
API Integration
```

DeepFace is responsible for creating the facial representation used by the recognition system, while the application's matching logic determines which registered person is the closest match.
