# DeepFace Watchlist: Web Frontend

A frontend in plain HTML, CSS and JavaScript for the DeepFace Watchlist & Recognition System (ArcFace + YuNet, euclidean_l2).

## Pages
| # | Page | File | Access |
|---|------|------|--------|
| 1 | Login (User / Admin) | `index.html` | public |
| 2 | Register Face | `register.html` | User or Admin |
| 3 | De-register Face | `deregister.html` | User or Admin |
| 4 | Inference (webcam + video) | `inference.html` | **public** |
| 5 | Admin Dashboard | `admin.html` | Admin only |

## Run it (LIVE mode, real face recognition)
`server.py` runs both the website and the DeepFace backend:

```powershell
cd C:\Users\Lecia\Downloads\deepface-web\deepface-web
C:\Users\Lecia\anaconda3\envs\myevv\python.exe server.py
# open http://localhost:8000
```

Registered photos are saved in **`faces/<name>/sample_01.jpg`** and so on. Names and statuses are in `watchlist.csv`, and the face vectors are in `arcface_embeddings_cache.pkl`. The cache is rebuilt by **Admin → Retrain**, or rebuilt automatically when you delete the file.

## Run it (DEMO mode, website only)
Webcam access needs `localhost` or HTTPS, so don't open the files with a `file://` path. Serve the folder instead:

```bash
cd deepface-web
python -m http.server 5500
# open http://localhost:5500
```

Demo accounts: **user / user123** and **admin / admin123**

## Demo mode vs live mode
By default the site runs in **DEMO MODE**. Data is kept in `localStorage`, and recognition results are **simulated**, because ArcFace doesn't run in the browser.

To connect your Python DeepFace backend, go to **Admin → Model Settings → Backend API URL** (for example `http://localhost:8000`) and save. The pill in the nav changes to **● LIVE API**.

## Backend API contract
Enable CORS on your backend (for FastAPI, use `CORSMiddleware`).

| Method | Path | Body (multipart) | Returns |
|---|---|---|---|
| POST | `/register` | `name`, `id_number`, `status`, `mode` (`new`/`existing`), `files[]` (jpg) | `{ "vectors": 42 }`: the person's total vector count |
| DELETE | `/persons/{name}` | – | `{}` |
| POST | `/recognize` | `frame` (jpg) | `{ "faces": [{ "box": [x,y,w,h], "name": "sh", "status": "Blacklisted", "similarity": 0.72 }] }` |
| POST | `/analyse-video` | `video`, `every_n` | `{ "frames": 101, "duration": 4.24, "annotated_url": "/files/out.mp4", "report": [{ "name", "id_number", "status", "first_seen", "last_seen", "detections", "best_similarity" }] }` |
| POST | `/retrain` | – | `{}` |

`status` is one of `Authorised`, `Blacklisted` or `Unknown`. `first_seen` and `last_seen` are in seconds. `best_similarity` is a percentage (for example 72.3).

## Structure
```
index.html  register.html  deregister.html  inference.html  admin.html
css/style.css
js/app.js         shared: storage, auth guard, nav, API bridge, toasts, modals
js/register.js    js/deregister.js    js/inference.js    js/admin.js
```

> Note: the login here is client-side only (sessionStorage), so it's fine for a demo but not real security. For production, move authentication to the backend.
