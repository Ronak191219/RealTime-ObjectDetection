# Refactoring Plan: Streamlit → FastAPI + React

## Overview

Refactor the current single-file Streamlit application into a separated frontend (React + Vite) and backend (FastAPI + Python) architecture while keeping the YOLOv8 ML logic identical.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        REACT FRONTEND                           │
│  http://localhost:5173                                         │
│                                                                 │
│  - Web UI with dark sci-fi theme                                │
│  - Live webcam video display                                    │
│  - Image upload & results display                               │
│  - Sidebar controls (mode, confidence, model)                  │
│  - Stats panel, detection list, logs                           │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST API + MJPEG Stream
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND                          │
│  http://localhost:8000                                         │
│                                                                 │
│  - YOLOv8 inference (unchanged from app.py)                    │
│  - /api/detect/live    → MJPEG video stream                    │
│  - /api/detect/image   → POST image, return JSON detections    │
│  - /api/logs           → GET CSV logs                          │
│  - /api/screenshots    → POST/GET screenshots                  │
│  - /api/models         → GET available models                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Plan

### Phase 1: Backend (FastAPI)

#### 1.1 Create Backend Project Structure
```
backend/
├── main.py              # FastAPI app entry point
├── requirements.txt     # Python dependencies
├── app/
│   ├── __init__.py
│   ├── model.py         # YOLOv8 loading (from app.py)
│   ├── camera.py        # CameraStream class (from app.py)
│   ├── detection.py     # draw_detections, draw_hud (from app.py)
│   ├── logging.py       # CSV logging functions (from app.py)
│   └── routes/
│       ├── __init__.py
│       ├── detect.py    # /api/detect/* endpoints
│       ├── logs.py      # /api/logs endpoint
│       └── screenshots.py # /api/screenshots endpoints
├── screenshots/         # Saved screenshots
└── logs/                # Detection CSV logs
```

#### 1.2 Implement Backend Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detect/live` | GET | MJPEG streaming endpoint for live webcam |
| `/api/detect/image` | POST | Upload image, return detections JSON |
| `/api/detect/start` | POST | Start camera capture |
| `/api/detect/stop` | POST | Stop camera capture |
| `/api/logs` | GET | Download detection logs CSV |
| `/api/screenshots` | POST | Save screenshot |
| `/api/screenshots` | GET | List saved screenshots |
| `/api/models` | GET | List available YOLOv8 models |

#### 1.3 Key Implementation Details

- **MJPEG Stream**: Use `StreamingResponse` with multipart/x-mixed-replace for live video
- **Camera Threading**: Keep camera in main thread or use proper thread-safe implementation
- **CORS**: Enable CORS for React frontend on port 5173
- **Static Files**: Serve screenshots folder as static files

---

### Phase 2: Frontend (React + Vite)

#### 2.1 Create Frontend Project Structure
```
frontend/
├── src/
│   ├── main.jsx
│   ├── App.jsx           # Main app component
│   ├── App.css           # Global styles (dark sci-fi theme)
│   ├── components/
│   │   ├── Sidebar.jsx   # Controls (mode, confidence, model)
│   │   ├── LiveWebcam.jsx # Live video + stats
│   │   ├── ImageUpload.jsx # Image upload + results
│   │   ├── DetectionList.jsx # Detected objects list
│   │   ├── MetricCards.jsx # FPS, Objects, Total stats
│   │   └── VideoStream.jsx # MJPEG video player
│   ├── services/
│   │   ├── api.js        # API client functions
│   │   └── stream.js     # MJPEG stream handler
│   └── hooks/
│       ├── useCamera.js  # Camera control hook
│       └── useDetection.js # Detection state hook
├── index.html
├── package.json
├── vite.config.js
└── .env                  # VITE_API_URL=http://localhost:8000
```

#### 2.2 Frontend Components

| Component | Responsibility |
|-----------|----------------|
| `App.jsx` | Route between Live/Image modes, manage global state |
| `Sidebar.jsx` | Mode selector, confidence slider, model dropdown, stats |
| `LiveWebcam.jsx` | Display MJPEG stream, start/stop/screenshot buttons |
| `ImageUpload.jsx` | File uploader, display original + annotated images |
| `VideoStream.jsx` | Handle MJPEG stream rendering |
| `DetectionList.jsx` | Show detected objects with confidence % |
| `MetricCards.jsx` | Display FPS, object count, total logged |

#### 2.3 API Integration

- Use `fetch` or `axios` for REST calls
- Use `HTMLImageElement` with `src` for MJPEG stream
- Handle loading states, errors gracefully

---

### Phase 3: Integration & Testing

#### 3.1 Run Both Services
```bash
# Terminal 1 - Backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

#### 3.2 Verify Features
- [ ] Live webcam detection works
- [ ] Image upload detection works
- [ ] Confidence threshold affects results
- [ ] Model switching works
- [ ] Screenshots save correctly
- [ ] Logs download works
- [ ] Stats update in real-time

---

## Dependencies

### Backend (Python)
```
fastapi
uvicorn
python-multipart
opencv-python
numpy
ultralytics
pillow
```

### Frontend (Node.js)
```
react
react-dom
vite
axios
```

---

## Migration Mapping (app.py → New Structure)

| Original (app.py) | New Location |
|-------------------|--------------|
| `@st.cache_resource load_model()` | `app/model.py` |
| `CameraStream` class | `app/camera.py` |
| `draw_detections()` | `app/detection.py` |
| `draw_hud()` | `app/detection.py` |
| `init_log()`, `log_detections()` | `app/logging.py` |
| CSS styles | `frontend/src/App.css` |
| `render_sidebar()` | `frontend/src/components/Sidebar.jsx` |
| `section_live_webcam()` | `frontend/src/components/LiveWebcam.jsx` |
| `section_image_upload()` | `frontend/src/components/ImageUpload.jsx` |
| `main()` | `frontend/src/App.jsx` |

---

## Notes

- **YOLOv8 stays in Python** — no ML code changes needed
- **FastAPI** chosen over Flask for better async support and automatic OpenAPI docs
- **MJPEG streaming** is the simplest way to transfer live video frames
- **React** provides more control over UI than Streamlit
- **CORS** must be configured to allow frontend-backend communication