"""
DeepFace Watchlist backend + web server.

Run:   python server.py
Open:  http://localhost:8000

Photos are saved in   faces/<name>/sample_01.jpg, sample_02.jpg, ...
Watchlist is saved in watchlist.csv  (name, status, id_number)
ArcFace embeddings:   arcface_embeddings_cache.pkl  (rebuilt automatically)
"""
import csv
import os
import pickle
import re
import shutil
import sys
import tempfile
import threading
import time

# DeepFace prints emoji warnings; don't crash on Windows consoles that can't show them
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

# ==========================================================
# SETTINGS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BASE_DIR, "faces")          # <- the photo folder
OUTPUT_DIR = os.path.join(BASE_DIR, "output")        # annotated videos
WATCHLIST_FILE = os.path.join(BASE_DIR, "watchlist.csv")
CACHE_FILE = os.path.join(BASE_DIR, "arcface_embeddings_cache.pkl")

MODEL_NAME = "ArcFace"
DETECTOR = "yunet"            # face detector for live frames (fast). Try "retinaface" for accuracy.
MATCH_THRESHOLD = 1.13        # euclidean_l2 distance; lower = stricter
PORT = 8000

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

lock = threading.RLock()
cache = {}  # name -> {"embeddings": np.ndarray, "status": str}



# ==========================================================
# WATCHLIST (CSV)
# ==========================================================
def load_watchlist():
    if not os.path.exists(WATCHLIST_FILE):
        return []
    with open(WATCHLIST_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{"name": r.get("name", ""), "status": r.get("status", "Authorised"),
             "id_number": r.get("id_number", "") or ""} for r in rows if r.get("name")]


def save_watchlist(rows):
    with open(WATCHLIST_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "status", "id_number"])
        w.writeheader()
        w.writerows(rows)


def safe_name(name):
    name = (name or "").strip()
    if not name or not re.fullmatch(r"[\w .\-']{1,60}", name) or name.strip(".") == "":
        raise HTTPException(400, "Name may only contain letters, numbers, spaces, . - _ '")
    return name


def person_photos(name):
    folder = os.path.join(FACES_DIR, name)
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))


# ==========================================================
# FACE HELPERS
# ==========================================================
def crop_face(img):
    """Crop the biggest face (with margin) for storage, using the SAME detector as live
    recognition so registered and live faces look alike. Keeps the whole image if none found."""
    try:
        found = DeepFace.extract_faces(img_path=img, detector_backend=DETECTOR,
                                       enforce_detection=True, align=False)
    except ValueError:  # no face found
        return img
    a = max((f["facial_area"] for f in found), key=lambda a: a["w"] * a["h"])
    x, y, w, h = a["x"], a["y"], a["w"], a["h"]
    mx, my = int(w * 0.2), int(h * 0.2)
    H, W = img.shape[:2]
    return img[max(0, y - my):min(H, y + h + my), max(0, x - mx):min(W, x + w + mx)]


def augment(img):
    h, w = img.shape[:2]
    c = (w // 2, h // 2)
    out = [img, cv2.flip(img, 1)]
    for a in (5, -5):
        out.append(cv2.warpAffine(img, cv2.getRotationMatrix2D(c, a, 1.0), (w, h), borderMode=cv2.BORDER_REPLICATE))
    out.append(cv2.convertScaleAbs(img, alpha=1.10, beta=12))
    out.append(cv2.convertScaleAbs(img, alpha=0.90, beta=-12))
    out.append(cv2.convertScaleAbs(img, alpha=1.20, beta=0))
    return out


def normalise(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def embed_crop(img):
    """Embedding of an already-cropped face photo."""
    try:
        r = DeepFace.represent(img_path=img, model_name=MODEL_NAME, detector_backend=DETECTOR,
                               enforce_detection=False, align=True)
        return normalise(r[0]["embedding"]) if r else None
    except Exception as e:
        print("embed error:", e)
        return None


def embed_person(name):
    vecs = []
    for f in person_photos(name):
        img = cv2.imread(os.path.join(FACES_DIR, name, f))
        if img is None:
            continue
        for a in augment(img):
            e = embed_crop(a)
            if e is not None:
                vecs.append(e)
    return np.array(vecs, dtype=np.float32)


def save_cache():
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


def retrain_all():
    global cache
    new = {}
    for row in load_watchlist():
        embs = embed_person(row["name"])
        if len(embs):
            new[row["name"]] = {"embeddings": embs, "status": row["status"]}
    with lock:
        cache = new
        save_cache()


def load_cache():
    global cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            return
        except Exception as e:
            print("cache load error:", e)
    retrain_all()


def match(emb, threshold):
    best, best_d = None, float("inf")
    with lock:
        for name, d in cache.items():
            if len(d["embeddings"]) == 0:
                continue
            dist = float(np.min(np.linalg.norm(d["embeddings"] - emb, axis=1)))
            if dist < best_d:
                best, best_d = name, dist
        status = cache[best]["status"] if best else None
    if best and best_d <= threshold:
        return best, status, max(0.0, 1.0 - best_d / 2.0)
    return "UNKNOWN", "Unknown", 0.0


def recognise(img, threshold=MATCH_THRESHOLD):
    try:
        reps = DeepFace.represent(img_path=img, model_name=MODEL_NAME, detector_backend=DETECTOR,
                                  enforce_detection=True, align=True)
    except ValueError:  # no face found
        return []
    faces = []
    for r in reps:
        a = r["facial_area"]
        name, status, sim = match(normalise(r["embedding"]), threshold)
        faces.append({"box": [a["x"], a["y"], a["w"], a["h"]], "name": name,
                      "status": status, "similarity": round(sim, 4)})
    return faces


def decode(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not read image")
    return img


# ==========================================================
# API
# ==========================================================
app = FastAPI(title="DeepFace Watchlist API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "detector": DETECTOR, "people": len(cache)}


@app.get("/persons")
def list_persons():
    out = []
    for row in load_watchlist():
        photos = person_photos(row["name"])
        out.append({
            "name": row["name"], "status": row["status"], "id_number": row["id_number"],
            "photos": [f"/faces/{row['name']}/{p}" for p in photos],
            "vectors": int(len(cache.get(row["name"], {}).get("embeddings", []))),
        })
    return out


@app.post("/register")
def register(name: str = Form(...), status: str = Form("Authorised"), id_number: str = Form(""),
             mode: str = Form("new"), files: list[UploadFile] = File(...)):
    name = safe_name(name)
    if status not in ("Authorised", "Blacklisted"):
        raise HTTPException(400, "status must be Authorised or Blacklisted")

    folder = os.path.join(FACES_DIR, name)
    if mode == "new" and os.path.isdir(folder):
        shutil.rmtree(folder)  # fresh registration replaces old photos
    os.makedirs(folder, exist_ok=True)

    existing = person_photos(name)
    nums = [int(m.group(1)) for f in existing if (m := re.match(r"sample_(\d+)", f))]
    n = max(nums, default=0)
    saved = 0
    for up in files:
        img = decode(up.file.read())
        n += 1
        cv2.imwrite(os.path.join(folder, f"sample_{n:02d}.jpg"), crop_face(img))
        saved += 1

    rows = [r for r in load_watchlist() if r["name"] != name]
    old = next((r for r in load_watchlist() if r["name"] == name), None)
    rows.append({"name": name, "status": status if mode == "new" or not old else old["status"],
                 "id_number": id_number or (old["id_number"] if old else "")})
    save_watchlist(rows)

    embs = embed_person(name)
    with lock:
        cache[name] = {"embeddings": embs, "status": rows[-1]["status"]}
        save_cache()
    print(f"Registered {name}: {saved} new photo(s), {len(embs)} vectors -> {folder}")
    return {"vectors": int(len(embs)), "photos": len(person_photos(name))}


@app.delete("/persons/{name}")
def delete_person(name: str):
    name = safe_name(name)
    save_watchlist([r for r in load_watchlist() if r["name"] != name])
    shutil.rmtree(os.path.join(FACES_DIR, name), ignore_errors=True)
    with lock:
        cache.pop(name, None)
        save_cache()
    return {}


@app.post("/recognize")
def recognize(frame: UploadFile = File(...), threshold: float = Form(MATCH_THRESHOLD)):
    return {"faces": recognise(decode(frame.file.read()), threshold)}


@app.post("/retrain")
def retrain():
    retrain_all()
    return {}


@app.post("/analyse-video")
def analyse_video(video: UploadFile = File(...), every_n: int = Form(3)):
    every_n = max(1, int(every_n))
    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(video.file, tmp)
        src = tmp.name
    try:
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_name = f"annotated_{int(time.time())}.mp4"
        writer = cv2.VideoWriter(os.path.join(OUTPUT_DIR, out_name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        colors = {"Blacklisted": (85, 45, 255), "Authorised": (143, 255, 31), "Unknown": (153, 139, 139)}
        ids = {r["name"]: r["id_number"] for r in load_watchlist()}
        tracks, idx, analysed, faces = {}, 0, 0, []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            t = idx / fps
            if idx % every_n == 0:
                faces = recognise(img)
                analysed += 1
                for f in faces:
                    tr = tracks.setdefault(f["name"], {"name": f["name"], "id_number": ids.get(f["name"], ""),
                                                       "status": f["status"], "first_seen": t, "last_seen": t,
                                                       "detections": 0, "best": 0.0})
                    tr["last_seen"] = t
                    tr["detections"] += 1
                    tr["best"] = max(tr["best"], f["similarity"])
            for f in faces:
                x, y, w, h = f["box"]
                c = colors.get(f["status"], colors["Unknown"])
                cv2.rectangle(img, (x, y), (x + w, y + h), c, 3)
                label = "UNKNOWN" if f["status"] == "Unknown" else f"{f['name']} {f['similarity'] * 100:.1f}%"
                cv2.putText(img, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
            writer.write(img)
            idx += 1
        cap.release()
        writer.release()
    finally:
        os.remove(src)
    report = [{**{k: v for k, v in r.items() if k != "best"}, "best_similarity": round(r["best"] * 100, 1)}
              for r in tracks.values()]
    return {"frames": analysed, "duration": idx / fps, "annotated_url": f"/files/{out_name}", "report": report}


# Tells the website it is being served by this backend -> LIVE mode automatically
@app.get("/js/config.js")
def config_js():
    return Response("window.DF_API_BASE = location.origin;", media_type="application/javascript")


app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")
app.mount("/faces", StaticFiles(directory=FACES_DIR), name="faces")
app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="site")


if __name__ == "__main__":
    import uvicorn
    print("Loading ArcFace model and face vectors (first run downloads the model)...")
    try:
        DeepFace.represent(img_path=np.zeros((112, 112, 3), np.uint8), model_name=MODEL_NAME,
                           detector_backend="skip", enforce_detection=False)
    except Exception as e:
        print("warm-up:", e)
    load_cache()
    print(f"Ready. {len(cache)} person(s) loaded. Photos folder: {FACES_DIR}")
    print(f"Open http://localhost:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT)
