from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Drone Face Recognition Information API",
    description="Receives the recognition result from Streamlit and exposes the latest result.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# LATEST RESULT - MEMORY ONLY
# No database and no JSON file.
# This is cleared when the API restarts.
# ============================================================

LATEST_IDENTIFICATION = {
    "person": "Unknown",
    "entity": "unknown",
    "confidence": 0.0,
    "picture_available": False,
}

LATEST_PICTURE_BYTES: Optional[bytes] = None
LATEST_PICTURE_CONTENT_TYPE = "image/jpeg"

LATEST_REGISTRATION = {
    "name": None,
    "list_type": None,
    "registered": False,
}


# ============================================================
# RESPONSE MODELS
# ============================================================

class IdentifyResponse(BaseModel):
    person: str = Field(..., example="Lecia")
    picture: str = Field(..., example="http://127.0.0.1:8000/identify/picture")
    entity: str = Field(..., example="good")
    confidence: float = Field(..., example=87.5)


class RegisterResponse(BaseModel):
    status: str = Field("registered", example="registered")


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Drone Face Recognition Information API is running",
        "endpoints": {
            "register": "POST /register",
            "identify_send": "POST /identify",
            "identify_view": "GET /identify",
            "identify_picture": "GET /identify/picture"
        }
    }


# ============================================================
# REGISTER
# ============================================================

@app.post("/register", response_model=RegisterResponse)
async def register(
    name: str = Form(...),
    list_type: str = Form(...),
    photo: UploadFile = File(...)
):
    global LATEST_REGISTRATION

    name = name.strip()
    list_type = list_type.strip().lower()

    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    if list_type not in {"white", "black"}:
        raise HTTPException(
            status_code=400,
            detail="list_type must be 'white' or 'black'."
        )

    # We receive the photo so the API can accept the registration request.
    # Nothing is saved to disk and no database is used.
    await photo.read()

    LATEST_REGISTRATION = {
        "name": name,
        "list_type": list_type,
        "registered": True
    }

    return {"status": "registered"}


# ============================================================
# IDENTIFY - RECEIVE RESULT FROM STREAMLIT
# ============================================================

@app.post("/identify", response_model=IdentifyResponse)
async def identify(
    person: str = Form(...),
    entity: str = Form(...),
    confidence: float = Form(...),
    picture: UploadFile = File(...)
):
    global LATEST_IDENTIFICATION
    global LATEST_PICTURE_BYTES
    global LATEST_PICTURE_CONTENT_TYPE

    person = person.strip() or "Unknown"
    entity = entity.strip().lower()

    if entity not in {"good", "bad", "unknown"}:
        raise HTTPException(
            status_code=400,
            detail="entity must be 'good', 'bad', or 'unknown'."
        )

    if not 0 <= confidence <= 100:
        raise HTTPException(
            status_code=400,
            detail="confidence must be between 0 and 100."
        )

    picture_bytes = await picture.read()

    LATEST_PICTURE_BYTES = picture_bytes
    LATEST_PICTURE_CONTENT_TYPE = picture.content_type or "image/jpeg"

    LATEST_IDENTIFICATION = {
        "person": person,
        "entity": entity,
        "confidence": round(float(confidence), 2),
        "picture_available": len(picture_bytes) > 0,
    }

    return {
        "person": person,
        "picture": "http://127.0.0.1:8000/identify/picture",
        "entity": entity,
        "confidence": round(float(confidence), 2),
    }


# ============================================================
# IDENTIFY - VIEW LATEST RESULT IN BROWSER
# ============================================================

@app.get("/identify")
def get_latest_identification():
    return {
        "person": LATEST_IDENTIFICATION["person"],
        "picture": "http://127.0.0.1:8000/identify/picture"
        if LATEST_IDENTIFICATION["picture_available"]
        else None,
        "entity": LATEST_IDENTIFICATION["entity"],
        "confidence": LATEST_IDENTIFICATION["confidence"],
    }


# ============================================================
# IDENTIFY - VIEW LATEST PICTURE
# ============================================================

@app.get("/identify/picture")
def get_latest_picture():
    if LATEST_PICTURE_BYTES is None:
        raise HTTPException(
            status_code=404,
            detail="No identification picture has been received yet."
        )

    return Response(
        content=LATEST_PICTURE_BYTES,
        media_type=LATEST_PICTURE_CONTENT_TYPE
    )


# ============================================================
# REGISTER - VIEW LATEST REGISTRATION (OPTIONAL)
# ============================================================

@app.get("/register")
def get_latest_registration():
    return LATEST_REGISTRATION
