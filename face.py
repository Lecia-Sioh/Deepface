import os
import pickle
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import requests
import time
import threading
import concurrent.futures
from deepface import DeepFace

# ==========================================================
# STREAMLIT CONFIGURATION (MUST BE FIRST STREAMLIT CALL)
# ==========================================================

st.set_page_config(
    page_title="DeepFace Watchlist System",
    layout="wide",
    page_icon="🛡️"
)

# ==========================================================
# SETTINGS & PATHS
# ==========================================================

DATABASE_PATH = "faces"
WATCHLIST_FILE = "watchlist.csv"
CACHE_FILE = "arcface_embeddings_cache.pkl"
# CACHE_FILE = "vggface_embeddings_cache.pkl"
MODEL_NAME = "ArcFace"
# MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "retinaface"
DETECTION_CONFIDENCE = 0.90   # Ignore DeepFace detections below this confidence
DISTANCE_METRIC = "euclidean_l2"
MATCH_THRESHOLD = 1.13   # Default reference: 1.17 for VGG-Face euclidean_l2 (was 1.13 for ArcFace)

# API
API_BASE_URL = "http://127.0.0.1:8000"
API_TIMEOUT = 5
API_SEND_INTERVAL = 0.5

# Performance Tuning
RECOGNITION_FRAME_INTERVAL = 3  # Perform full DeepFace recognition every N frames

os.makedirs(DATABASE_PATH, exist_ok=True)

if not os.path.exists(WATCHLIST_FILE):
    df = pd.DataFrame(columns=["name", "status"])
    df.to_csv(WATCHLIST_FILE, index=False)

# Thread pool for non-blocking background API calls
api_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


# ==========================================================
# THREADED WEBCAM CAPTURE FOR LAG-FREE LIVE FEED
# ==========================================================

class ThreadedCamera:
    """
    Continuously captures camera frames in a background thread
    to prevent frame buffer queuing lag.
    """
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.01)  # Yield CPU briefly

    def read(self):
        with self.read_lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def isOpened(self):
        return self.cap.isOpened()

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()


# ==========================================================
# DEEPFACE MODEL WARM-UP (PRELOAD WEIGHTS INTO RAM)
# ==========================================================

@st.cache_resource
def warmup_deepface_model():
    try:
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.represent(
            img_path=dummy_img,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False
        )
    except Exception as e:
        print("Warmup notice:", e)

warmup_deepface_model()


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def send_register_to_api(name, status_value, image_path):
    """Send registration information to the API."""
    list_type = "white" if status_value == "Authorised" else "black"

    try:
        with open(image_path, "rb") as f:
            files = {
                "photo": (os.path.basename(image_path), f, "image/jpeg")
            }
            data = {
                "name": name,
                "list_type": list_type
            }
            response = requests.post(
                f"{API_BASE_URL}/register",
                data=data,
                files=files,
                timeout=API_TIMEOUT
            )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"API /register failed: {e}"}
    except Exception as e:
        return {"error": f"API /register error: {e}"}


def send_identify_to_api(person, picture, entity, confidence):
    """
    Send the identification RESULT to the API.
    Executed asynchronously so main UI stream never lags.
    """
    try:
        ok, encoded = cv2.imencode(".jpg", picture)
        if not ok:
            return {"error": "Could not encode identification picture."}

        files = {
            "picture": ("identified_face.jpg", encoded.tobytes(), "image/jpeg")
        }
        data = {
            "person": person,
            "entity": entity,
            "confidence": str(float(confidence))
        }

        response = requests.post(
            f"{API_BASE_URL}/identify",
            data=data,
            files=files,
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException as e:
        return {"error": f"API /identify failed: {e}"}
    except Exception as e:
        return {"error": f"API /identify error: {e}"}


def send_identify_async(person, picture, entity, confidence):
    """Submits API identification post to non-blocking background thread pool."""
    def _job():
        send_identify_to_api(person, picture, entity, confidence)
    api_executor.submit(_job)


def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            return pd.read_csv(WATCHLIST_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=["name", "status"])


def get_registered_image_path(name, data=None):
    """Retrieve the primary registered face image path for a person."""
    if data and isinstance(data, dict):
        img_path = data.get("image_path")
        if img_path and os.path.exists(img_path):
            return img_path

    person_folder = os.path.join(DATABASE_PATH, name)
    if os.path.isdir(person_folder):
        files = sorted([
            f for f in os.listdir(person_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        if files:
            return os.path.join(person_folder, files[0])

    return None


def update_person_status(name, new_status, cache=None):
    """Update a person's status in watchlist.csv and the cache file."""
    df = load_watchlist()
    if name in df["name"].values:
        df.loc[df["name"] == name, "status"] = new_status
    else:
        new_row = pd.DataFrame({"name": [name], "status": [new_status]})
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(WATCHLIST_FILE, index=False)

    if cache is not None and name in cache:
        cache[name]["status"] = new_status
        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(cache, f)
        except Exception as e:
            print("Failed to save updated cache:", e)


def quick_blacklist_live_face(person_name, face_crop, cache=None):
    """Register a new face directly from the live feed and blacklist them."""
    person_name = person_name.strip()
    if not person_name:
        return False, "Person name cannot be empty."

    person_folder = os.path.join(DATABASE_PATH, person_name)
    os.makedirs(person_folder, exist_ok=True)
    sample_path = os.path.join(person_folder, "sample_01.jpg")
    cv2.imwrite(sample_path, face_crop)

    update_person_status(person_name, "Blacklisted", cache)
    new_cache = retrain_watchlist_cache()
    if cache is not None:
        cache.clear()
        cache.update(new_cache)

    return True, f"Successfully registered and blacklisted {person_name}!"


def detect_faces(img, min_size=(50, 50)):
    """Detect faces with DeepFace and return a list of (x, y, w, h) boxes."""
    if img is None or img.size == 0:
        return []

    try:
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=False
        )
    except Exception as e:
        print("Face detection error:", e)
        return []

    h_img, w_img = img.shape[:2]
    boxes = []
    for face in faces:
        # With enforce_detection=False, DeepFace returns the whole image
        # with confidence 0 when no face is found - skip those.
        if face.get("confidence", 0) < DETECTION_CONFIDENCE:
            continue

        area = face["facial_area"]
        x, y, w, h = int(area["x"]), int(area["y"]), int(area["w"]), int(area["h"])
        if w < min_size[0] or h < min_size[1]:
            continue
        if w >= w_img and h >= h_img:
            continue

        boxes.append((x, y, w, h))

    return boxes


def crop_face_with_margin(img, box, margin=0.10):
    h_img, w_img = img.shape[:2]
    x, y, w, h = box

    margin_x = int(w * margin)
    margin_y = int(h * margin)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w_img, x + w + margin_x)
    y2 = min(h_img, y + h + margin_y)

    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else img[y:y+h, x:x+w]


def generate_face_augmentations(face_img):
    augmented = [face_img]
    h, w = face_img.shape[:2]
    center = (w // 2, h // 2)

    # Horizontal Flip
    augmented.append(cv2.flip(face_img, 1))

    # Rotations
    for angle in [5, -5]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        augmented.append(rotated)

    # Brightness adjustment
    brighter = cv2.convertScaleAbs(face_img, alpha=1.10, beta=12)
    augmented.append(brighter)

    darker = cv2.convertScaleAbs(face_img, alpha=0.90, beta=-12)
    augmented.append(darker)

    # High Contrast
    high_contrast = cv2.convertScaleAbs(face_img, alpha=1.20, beta=0)
    augmented.append(high_contrast)

    return augmented


def extract_face_embedding(face_img, fast_mode=True):
    if face_img is None or face_img.size == 0:
        return None

    try:
        # If fast_mode=True (for already cropped face bounding boxes),
        # using backend="skip" prevents running RetinaFace a SECOND time on the small crop,
        # which eliminates 95% of live feed lag.
        backend = "skip" if fast_mode else DETECTOR_BACKEND

        results = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend=backend,
            enforce_detection=False
        )

        if results and len(results) > 0:
            raw_emb = np.array(results[0]["embedding"], dtype=np.float32)
            norm = np.linalg.norm(raw_emb)
            if norm > 0:
                return raw_emb / norm
            return raw_emb
    except Exception as e:
        print("Embedding extraction error:", e)

    return None


def retrain_watchlist_cache(progress_bar=None):
    watchlist_df = load_watchlist()
    cache = {}
    total_people = len(watchlist_df)

    for i, (_, row) in enumerate(watchlist_df.iterrows()):
        name = row["name"]
        status = row["status"]

        person_folder = os.path.join(DATABASE_PATH, name)
        if not os.path.isdir(person_folder):
            continue

        image_files = sorted([
            f for f in os.listdir(person_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        embeddings_list = []

        for img_file in image_files:
            img_path = os.path.join(person_folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            aug_imgs = generate_face_augmentations(img)
            for aug in aug_imgs:
                emb = extract_face_embedding(aug)
                if emb is not None:
                    embeddings_list.append(emb)

        if len(embeddings_list) > 0:
            cache[name] = {
                "embeddings": np.array(embeddings_list, dtype=np.float32),
                "status": status,
                "image_path": os.path.join(person_folder, image_files[0]),
                "aug_count": len(embeddings_list)
            }

        if progress_bar is not None and total_people > 0:
            progress_bar.progress((i + 1) / total_people)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

    return cache


def load_embeddings_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            if isinstance(cache, dict):
                return cache
        except Exception as e:
            print("Embedding cache load error:", e)

    return retrain_watchlist_cache()


def register_face(name, status, image_inputs, progress_bar=None):
    person_folder = os.path.join(DATABASE_PATH, name)
    os.makedirs(person_folder, exist_ok=True)

    for old_file in os.listdir(person_folder):
        old_path = os.path.join(person_folder, old_file)
        if os.path.isfile(old_path):
            os.remove(old_path)

    if not isinstance(image_inputs, list):
        image_inputs = [image_inputs]

    saved_paths = []

    for idx, img_input in enumerate(image_inputs, start=1):
        if isinstance(img_input, np.ndarray):
            img = img_input
        else:
            file_bytes = np.frombuffer(img_input.getbuffer(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            continue

        detected = detect_faces(img)
        if len(detected) > 0:
            best_face = max(detected, key=lambda f: f[2] * f[3])
            face_crop = crop_face_with_margin(img, best_face)
        else:
            face_crop = img

        sample_path = os.path.join(person_folder, f"sample_{idx:02d}.jpg")
        cv2.imwrite(sample_path, face_crop)
        saved_paths.append(sample_path)

    if len(saved_paths) == 0:
        raise ValueError("Could not decode or detect faces in uploaded images.")

    df = load_watchlist()
    df = df[df["name"] != name]
    new_entry = pd.DataFrame({"name": [name], "status": [status]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(WATCHLIST_FILE, index=False)

    cache = retrain_watchlist_cache(progress_bar)
    aug_count = cache.get(name, {}).get("aug_count", 0)
    return saved_paths, aug_count


def delete_face(name):
    df = load_watchlist()
    df = df[df["name"] != name]
    df.to_csv(WATCHLIST_FILE, index=False)

    person_folder = os.path.join(DATABASE_PATH, name)
    if os.path.exists(person_folder):
        for file in os.listdir(person_folder):
            file_path = os.path.join(person_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(person_folder)

    retrain_watchlist_cache()


def match_face_embedding(query_emb, cache, threshold=1.17):
    if query_emb is None or not cache:
        return None, None, float("inf"), 0.0

    best_name = None
    best_status = None
    min_dist = float("inf")

    query_norm = np.linalg.norm(query_emb)
    if query_norm > 0:
        query_emb = query_emb / query_norm

    for name, data in cache.items():
        embeddings = data.get("embeddings", [])
        status = data.get("status", "Unknown")

        if len(embeddings) == 0:
            continue

        # Fast Euclidean L2 Distance calculation
        dists = np.linalg.norm(embeddings - query_emb, axis=1)
        person_min_dist = float(np.min(dists))

        if person_min_dist < min_dist:
            min_dist = person_min_dist
            best_name = name
            best_status = status

    if best_name is not None and min_dist <= threshold:
        similarity_pct = max(0.0, min(100.0, (1.0 - min_dist / 2.0) * 100.0))
        return best_name, best_status, min_dist, similarity_pct

    return None, None, min_dist, 0.0


# ==========================================================
# STREAMLIT UI SETUP
# ==========================================================

st.title("🛡️ DeepFace Watchlist & Recognition System")
st.write(f"{MODEL_NAME} embeddings with {DETECTOR_BACKEND} face detection and non-blocking streaming.")

option = st.sidebar.selectbox(
    "Select Function",
    ["Register Face", "De-register Face", "Live Recognition"]
)


# ==========================================================
# 1. REGISTER FACE
# ==========================================================

if option == "Register Face":
    st.header("👤 Register New Face")

    col1, col2 = st.columns([1, 1])

    with col1:
        name = st.text_input("Person Name", placeholder="e.g. John Doe")
        status = st.selectbox(
            "Watchlist Status",
            ["Authorised", "Blacklisted"]
        )

    with col2:
        reg_mode = st.radio(
            "Registration Method",
            ["Upload Photo(s)", "Take Snapshot with Webcam"]
        )

    image_to_process = None

    if reg_mode == "Upload Photo(s)":
        uploaded_files = st.file_uploader(
            "Upload face photo(s) (Different angles/lighting recommended)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        if uploaded_files:
            image_to_process = uploaded_files
    else:
        camera_photo = st.camera_input("Take a photo of the face")
        if camera_photo is not None:
            image_to_process = [camera_photo]

    if st.button("Register & Train Model", type="primary"):
        if not name.strip():
            st.error("Please enter the person's name.")
        elif not image_to_process:
            st.error("Please upload at least one image or capture a webcam snapshot.")
        else:
            progress = st.progress(0.0)
            status_text = st.empty()
            status_text.info(f"Detecting faces, generating augmentations, and updating {MODEL_NAME} embeddings...")

            try:
                saved_paths, aug_count = register_face(
                    name.strip(),
                    status,
                    image_to_process,
                    progress
                )

                progress.empty()
                status_text.empty()

                st.success(f"✅ **{name}** registered successfully as **{status}**!")
                st.info(
                    f"Saved {len(saved_paths)} sample image(s) and trained {aug_count} augmented vector embeddings."
                )

                api_result = send_register_to_api(
                    name.strip(),
                    status,
                    saved_paths[0]
                )

                if "error" in api_result:
                    st.warning(f"API /register notice: {api_result['error']}")
                else:
                    st.success(
                        f"📡 API /register → "
                        f"{api_result.get('status', 'registered')}"
                    )

                if len(saved_paths) > 0:
                    saved_img = cv2.imread(saved_paths[0])
                    if saved_img is not None:
                        st.subheader("Augmented Training Vector Previews:")
                        aug_samples = generate_face_augmentations(saved_img)
                        titles = [
                            "Original", "Flipped", "+5° Rotate",
                            "-5° Rotate", "Brighter", "Darker", "High Contrast"
                        ]
                        aug_cols = st.columns(len(aug_samples))
                        for i, sample in enumerate(aug_samples):
                            with aug_cols[i]:
                                sample_rgb = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
                                st.image(sample_rgb, caption=titles[i], use_container_width=True)

            except Exception as e:
                progress.empty()
                status_text.empty()
                st.error(f"Registration Error: {e}")


# ==========================================================
# 2. DE-REGISTER FACE
# ==========================================================

elif option == "De-register Face":
    st.header("🗑️ De-register Face")

    df = load_watchlist()

    if len(df) == 0:
        st.info("No registered faces found in watchlist database.")
    else:
        st.subheader("Currently Registered Persons:")
        st.write(df.to_html(index=False), unsafe_allow_html=True)

        selected_name = st.selectbox(
            "Select person to remove",
            df["name"].tolist()
        )

        if st.button("Remove Person", type="primary"):
            with st.spinner("Removing person and retraining embedding cache..."):
                try:
                    delete_face(selected_name)
                    st.success(f"Removed **{selected_name}** from watchlist.")
                    st.rerun()
                except Exception as e:
                    st.error(f"De-registration error: {e}")


# ==========================================================
# ==========================================================
# 3. LIVE RECOGNITION
# ==========================================================

elif option == "Live Recognition":
    st.header("📹 Live Face Recognition & Watchlist Monitoring")

    cache = load_embeddings_cache()

    if not cache:
        st.warning("No faces registered yet. Please register at least one face first.")
    else:
        col_feed, col_side = st.columns([3, 2])

        with col_feed:
            col_t1, col_t2 = st.columns([2, 1])

            with col_t1:
                st.caption(
                    f"Model: {MODEL_NAME} | Metric: {DISTANCE_METRIC} | "
                    f"Threshold: {MATCH_THRESHOLD:.2f}"
                )

            with col_t2:
                start_camera = st.toggle(
                    "Start Camera Feed",
                    value=False,
                    key="start_camera_feed"
                )

            frame_placeholder = st.empty()
            status_placeholder = st.empty()

            # ------------------------------------------------------
            # LIVE-FEED REGISTRATION AREA
            # ------------------------------------------------------
            # This does NOT upload a new photo.
            # It uses the unknown face captured directly from the camera.
            st.markdown("##### 🚨 Register Unknown Person from Live Feed")

            # ==============================================================
            # STEP 1: PROCESS PENDING REGISTRATION (before camera restarts)
            # ==============================================================
            if st.session_state.get("pending_blacklist_register"):
                pending_name = st.session_state.get("pending_blacklist_name", "")
                pending_face = st.session_state.get("latest_unknown_face")

                # Clear the pending flag immediately
                st.session_state["pending_blacklist_register"] = False

                if pending_face is not None and pending_name.strip():
                    with st.spinner(
                        "Saving face and updating DeepFace embeddings..."
                    ):
                        ok, msg = quick_blacklist_live_face(
                            pending_name,
                            pending_face,
                            cache
                        )

                    if ok:
                        st.session_state["latest_unknown_face"] = None
                        st.session_state["pending_blacklist_name"] = ""
                        st.success(
                            f"🚨 {pending_name} is now BLACKLISTED and "
                            f"registered in the system!"
                        )
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.error(
                        "No unknown face available to register. "
                        "Make sure an unknown person is visible on camera first."
                    )

            # ==============================================================
            # STEP 2: ALWAYS SHOW THE REGISTRATION FORM
            # ==============================================================
            # The face preview and form are rendered BEFORE the camera loop.
            # - unknown_face_preview: updated live from inside the loop
            # - text_input + button: always visible so the user can type
            #   a name and click Register at any time, which breaks the
            #   loop via Streamlit rerun.
            # ==============================================================
            reg_c1, reg_c2 = st.columns([1, 2])

            with reg_c1:
                unknown_face_preview = st.empty()
                # Show any previously captured face from session_state
                latest_unknown = st.session_state.get("latest_unknown_face")
                if latest_unknown is not None:
                    unknown_face_preview.image(
                        cv2.cvtColor(latest_unknown, cv2.COLOR_BGR2RGB),
                        caption="Unknown face (live)",
                        width=160
                    )
                else:
                    unknown_face_preview.info("No face captured yet")

            with reg_c2:
                st.warning(
                    "⚠️ When an **UNKNOWN** person appears on camera, "
                    "their face will be captured here automatically. "
                    "Enter a name and click Register to blacklist them."
                )

                live_name = st.text_input(
                    "Person Name",
                    placeholder="e.g. Target_01",
                    key="live_blacklist_name"
                )

                if st.button(
                    "🔴 Register This Person as Blacklisted",
                    type="primary",
                    use_container_width=True,
                    key="register_unknown_live"
                ):
                    latest_check = st.session_state.get("latest_unknown_face")
                    if latest_check is None:
                        st.error(
                            "No unknown face captured yet. "
                            "Show an unregistered person to the camera first."
                        )
                    elif not live_name.strip():
                        st.error("Please enter a name first.")
                    else:
                        # Set pending flag — processed in STEP 1 on next rerun
                        st.session_state["pending_blacklist_register"] = True
                        st.session_state["pending_blacklist_name"] = live_name.strip()
                        st.rerun()

            # ------------------------------------------------------
            # CAMERA
            # ------------------------------------------------------
            if not start_camera:
                frame_placeholder.info(
                    "📷 Camera is turned off. Toggle "
                    "'Start Camera Feed' to start live face recognition."
                )
                status_placeholder.empty()

            else:
                camera = ThreadedCamera(src=0, width=640, height=480)
                camera.start()

                time.sleep(0.3)

                if not camera.isOpened():
                    status_placeholder.error(
                        "Unable to access webcam. Please check camera connection and permissions."
                    )
                else:
                    status_placeholder.info(
                        "📷 Camera active. Fast face scanning active..."
                    )

                    frame_count = 0

                    # Cache recent bounding boxes.
                    # (x, y, w, h, label, color)
                    cached_face_results = []

                    try:
                        while start_camera:
                            ret, frame = camera.read()

                            if not ret or frame is None:
                                time.sleep(0.01)
                                continue

                            frame_count += 1
                            display_frame = frame.copy()

                            # ------------------------------------------
                            # DeepFace recognition every N frames
                            # ------------------------------------------
                            if frame_count % RECOGNITION_FRAME_INTERVAL == 0:
                                detected_faces = detect_faces(frame)
                                new_results = []

                                for (x, y, w, h) in detected_faces:
                                    x, y, w, h = int(x), int(y), int(w), int(h)

                                    face_crop = crop_face_with_margin(
                                        frame,
                                        (x, y, w, h)
                                    )

                                    if face_crop.size == 0:
                                        continue

                                    # Save this exact face crop.
                                    # If DeepFace says Unknown, this is the
                                    # face that will be registered later.
                                    live_emb = extract_face_embedding(face_crop)

                                    best_name, best_status, min_dist, conf_pct = (
                                        match_face_embedding(
                                            live_emb,
                                            cache,
                                            threshold=MATCH_THRESHOLD
                                        )
                                    )

                                    if best_name is None:
                                        # UNKNOWN
                                        color = (0, 255, 255)
                                        label = "UNKNOWN"

                                        # Save the face crop to session_state
                                        # for registration.
                                        st.session_state[
                                            "latest_unknown_face"
                                        ] = face_crop.copy()

                                        # Live-update the face preview in
                                        # the registration panel so the user
                                        # can see it WITHOUT stopping the camera.
                                        try:
                                            unknown_face_preview.image(
                                                cv2.cvtColor(
                                                    face_crop,
                                                    cv2.COLOR_BGR2RGB
                                                ),
                                                caption="Unknown face (live)",
                                                width=160
                                            )
                                        except Exception:
                                            pass

                                        status_placeholder.warning(
                                            "⚠️ UNKNOWN PERSON detected! "
                                            "Type a name below and click "
                                            "'Register' to blacklist them."
                                        )

                                    elif best_status == "Blacklisted":
                                        # BLACKLISTED
                                        color = (0, 0, 255)
                                        label = (
                                            f"BLACKLISTED: {best_name} "
                                            f"({conf_pct:.0f}%)"
                                        )

                                        status_placeholder.error(
                                            f"🚨 BLACKLISTED: {best_name} | "
                                            f"Similarity: {conf_pct:.1f}%"
                                        )

                                    else:
                                        # AUTHORISED
                                        color = (0, 255, 0)
                                        label = (
                                            f"AUTHORISED: {best_name} "
                                            f"({conf_pct:.0f}%)"
                                        )

                                        status_placeholder.success(
                                            f"✅ AUTHORISED: {best_name} | "
                                            f"Similarity: {conf_pct:.1f}%"
                                        )

                                    new_results.append(
                                        (x, y, w, h, label, color)
                                    )

                                cached_face_results = new_results

                            # ------------------------------------------
                            # Draw bounding boxes ONLY.
                            # No targeting circle.
                            # ------------------------------------------
                            for (x, y, w, h, label, color) in cached_face_results:
                                cv2.rectangle(
                                    display_frame,
                                    (x, y),
                                    (x + w, y + h),
                                    color,
                                    3
                                )

                                y_text = max(y - 10, 25)

                                (text_w, text_h), _ = cv2.getTextSize(
                                    label,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    2
                                )

                                cv2.rectangle(
                                    display_frame,
                                    (x, y_text - text_h - 4),
                                    (x + text_w + 6, y_text + 4),
                                    color,
                                    -1
                                )

                                text_color = (
                                    (0, 0, 0)
                                    if color == (0, 255, 255)
                                    else (255, 255, 255)
                                )

                                cv2.putText(
                                    display_frame,
                                    label,
                                    (x + 3, y_text),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    text_color,
                                    2
                                )

                            frame_placeholder.image(
                                cv2.cvtColor(
                                    display_frame,
                                    cv2.COLOR_BGR2RGB
                                ),
                                channels="RGB",
                                use_container_width=True
                            )

                            time.sleep(0.01)

                    finally:
                        camera.stop()

        # ----------------------------------------------------------
        # RIGHT SIDE: REGISTERED PEOPLE
        # ----------------------------------------------------------
        with col_side:
            st.subheader("📋 Registered Watchlist Profiles")

            with st.container(height=500):
                for idx, (p_name, p_data) in enumerate(cache.items()):
                    p_img_path = get_registered_image_path(p_name, p_data)
                    p_status = p_data.get("status", "Unknown")

                    badge = (
                        "🔴 Blacklisted"
                        if p_status == "Blacklisted"
                        else "🟢 Authorised"
                    )

                    aug_c = p_data.get(
                        "aug_count",
                        len(p_data.get("embeddings", []))
                    )

                    c_img, c_desc = st.columns([1, 2])

                    with c_img:
                        if p_img_path and os.path.exists(p_img_path):
                            st.image(p_img_path, width=90)
                        else:
                            st.caption("No photo")

                    with c_desc:
                        st.markdown(f"**{p_name}**")
                        st.caption(badge)
                        st.caption(f"{aug_c} vectors")

                    if idx < len(cache) - 1:
                        st.divider()

