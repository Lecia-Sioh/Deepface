import os
import pickle
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import requests
import time
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
CACHE_FILE = "embeddings_cache.pkl"
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

# API
API_BASE_URL = "http://127.0.0.1:8000"
API_TIMEOUT = 5
API_SEND_INTERVAL = 0.5

os.makedirs(DATABASE_PATH, exist_ok=True)

if not os.path.exists(WATCHLIST_FILE):
    df = pd.DataFrame(columns=["name", "status"])
    df.to_csv(WATCHLIST_FILE, index=False)


# ==========================================================
# HAAR CASCADE INITIALIZATION
# ==========================================================

def load_face_cascade():
    if os.path.exists(CASCADE_FILENAME):
        cascade = cv2.CascadeClassifier(CASCADE_FILENAME)
        if not cascade.empty():
            return cascade, CASCADE_FILENAME

    opencv_path = os.path.join(cv2.data.haarcascades, CASCADE_FILENAME)
    if os.path.exists(opencv_path):
        cascade = cv2.CascadeClassifier(opencv_path)
        if not cascade.empty():
            return cascade, opencv_path

    return None, None


face_cascade, CASCADE_PATH = load_face_cascade()

if face_cascade is None:
    st.error(
        f"Could not load Haar Cascade XML file ({CASCADE_FILENAME}). "
        "Please ensure OpenCV is properly installed."
    )
    st.stop()


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
    Recognition remains in Streamlit.
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


def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            return pd.read_csv(WATCHLIST_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=["name", "status"])


def detect_faces(img, scale_factor=1.1, min_neighbors=5, min_size=(50, 50)):
    if img is None or img.size == 0:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )


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


def extract_face_embedding(face_img):
    if face_img is None or face_img.size == 0:
        return None

    try:
        results = DeepFace.represent(
            img_path=face_img,
            model_name="Facenet512",
            detector_backend="skip",
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


def match_face_embedding(query_emb, cache, threshold=0.38):
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

        sims = np.dot(embeddings, query_emb)
        dists = 1.0 - sims
        person_min_dist = float(np.min(dists))

        if person_min_dist < min_dist:
            min_dist = person_min_dist
            best_name = name
            best_status = status

    if best_name is not None and min_dist <= threshold:
        # Cosine Similarity percentage: (1.0 - min_dist) * 100
        confidence_pct = max(0.0, min(100.0, (1.0 - min_dist) * 100.0))
        return best_name, best_status, min_dist, confidence_pct

    return None, None, min_dist, 0.0


# ==========================================================
# STREAMLIT UI SETUP
# ==========================================================

st.title("🛡️ DeepFace Watchlist & Recognition System")
st.write("Augmented FaceNet512 Vector Embeddings with Direct 1-NN Blacklist Verification.")

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
            status_text.info("Detecting faces, generating augmentations, and updating FaceNet512 embeddings...")

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
                    st.warning(f"API /register failed: {api_result['error']}")
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
        # Render HTML table to avoid PyArrow DLL load issues on Windows/Anaconda
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
# 3. LIVE RECOGNITION
# ==========================================================

elif option == "Live Recognition":
    st.header("📹 Live Face Recognition & Watchlist Monitoring")

    cache = load_embeddings_cache()

    if not cache:
        st.warning("No faces registered yet. Please register at least one face first.")
    else:
        st.subheader("Watchlist Database Status")

        status_cols = st.columns(min(len(cache), 4))
        for i, (name, data) in enumerate(cache.items()):
            badge = "🔴 BLACKLISTED" if data["status"] == "Blacklisted" else "🟢 AUTHORISED"
            aug_c = data.get("aug_count", len(data.get("embeddings", [])))
            with status_cols[i % 4]:
                st.metric(label=f"{name}", value=badge, delta=f"{aug_c} Training Vectors")

        st.divider()

        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            MATCH_THRESHOLD = 0.8 
            #shang changed
            # MATCH_THRESHOLD = st.slider(
            #     "Recognition Cosine Distance Threshold (Lower = Stricter)",
            #     min_value=0.20,
            #     max_value=0.60,
            #     value=0.38,
            #     step=0.01,
            #     help="Recommended: 0.35 to 0.40. Face matches with cosine distance below this threshold are recognized."
            # )
        with col_t2:
            start_camera = st.checkbox("Start Camera Feed", value=False)

        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        if start_camera:
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            last_api_send = 0.0

            if not camera.isOpened():
                st.error("Unable to access webcam. Please check camera connection and permissions.")
            else:
                status_placeholder.info("📷 Camera active. Scanning for faces...")

                while start_camera:
                    ret, frame = camera.read()
                    if not ret:
                        status_placeholder.error("Failed to capture frame from webcam.")
                        break

                    display_frame = frame.copy()
                    detected_faces = detect_faces(frame)

                    if len(detected_faces) == 0:
                        status_placeholder.info("📷 Camera active. Scanning for faces...")

                    for (x, y, w, h) in detected_faces:
                        x = max(0, int(x))
                        y = max(0, int(y))
                        w = int(w)
                        h = int(h)

                        face_crop = crop_face_with_margin(frame, (x, y, w, h))
                        if face_crop.size == 0:
                            continue

                        live_emb = extract_face_embedding(face_crop)

                        best_name, best_status, min_dist, conf_pct = match_face_embedding(
                            live_emb,
                            cache,
                            threshold=MATCH_THRESHOLD
                        )

                        if best_name is not None and best_status == "Blacklisted":
                            color = (0, 0, 255)  # Red for Blacklisted
                            label = f"TARGET: {best_name} ({conf_pct:.0f}%)"
                            status_placeholder.error(
                                f"🚨 TARGET DETECTED: {best_name} (Blacklisted) | Distance: {min_dist:.3f} | Similarity: {conf_pct:.1f}%"
                            )
                        elif best_name is not None and best_status == "Authorised":
                            color = (0, 255, 0)  # Green for Authorised
                            label = f"AUTHORISED: {best_name} ({conf_pct:.0f}%)"
                            status_placeholder.success(
                                f"✅ AUTHORISED: {best_name} | Distance: {min_dist:.3f} | Similarity: {conf_pct:.1f}%"
                            )
                        else:
                            color = (0, 255, 255)  # Yellow for Unknown
                            label = "UNKNOWN"
                            status_placeholder.warning(
                                f"⚠️ Unknown face detected. Min Distance: {min_dist:.3f} (Threshold: {MATCH_THRESHOLD})"
                            )

                        # --------------------------------------------------
                        # SEND IDENTIFY RESULT TO API
                        # --------------------------------------------------
                        entity_for_api = (
                            "bad"
                            if best_name is not None and best_status == "Blacklisted"
                            else "good"
                            if best_name is not None and best_status == "Authorised"
                            else "unknown"
                        )

                        person_for_api = (
                            best_name if best_name is not None else "Unknown"
                        )
                        confidence_for_api = (
                            conf_pct if best_name is not None else 0.0
                        )

                        now = time.time()
                        if now - last_api_send >= API_SEND_INTERVAL:
                            api_result = send_identify_to_api(
                                person=person_for_api,
                                picture=face_crop,
                                entity=entity_for_api,
                                confidence=confidence_for_api
                            )

                            last_api_send = now

                            if "error" in api_result:
                                print(api_result["error"])
                            else:
                                st.caption(
                                    f"📡 API /identify → "
                                    f"{api_result.get('person', 'Unknown')} | "
                                    f"{api_result.get('entity', 'unknown')} | "
                                    f"{api_result.get('confidence', 0)}%"
                                )

                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)

                        y_text = max(y - 10, 25)
                        (text_w, text_h), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        cv2.rectangle(
                            display_frame,
                            (x, y_text - text_h - 4),
                            (x + text_w + 6, y_text + 4),
                            color,
                            -1
                        )

                        text_color = (0, 0, 0) if color == (0, 255, 255) else (255, 255, 255)
                        cv2.putText(
                            display_frame,
                            label,
                            (x + 3, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            text_color,
                            2
                        )

                    display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(
                        display_frame_rgb,
                        channels="RGB",
                        use_container_width=True
                    )

                camera.release()
