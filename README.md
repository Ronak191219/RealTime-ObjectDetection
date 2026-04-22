# 🎯 Real-Time Object Detection & Recognition System

> **YOLOv8 · OpenCV · Streamlit · WebRTC**  
> Detects 80 COCO object classes live from webcam or uploaded images with a polished dark UI.

---

## 📁 Project Structure

```
object_detection_system/
│
├── app.py                  ← Main application (single-file, fully commented)
├── requirements.txt        ← All Python dependencies
├── README.md               ← This file
│
├── screenshots/            ← Auto-created: saved frame captures
│   └── detection_YYYYMMDD_HHMMSS.jpg
│
└── logs/
    └── detection_logs.csv  ← Auto-created: timestamped detection history
```

---

## ⚙️ Setup Instructions

### 1 — Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | 3.10 +  |
| pip         | latest  |
| Webcam      | USB / built-in |

> **GPU (optional):** The app runs on CPU by default. For faster inference with NVIDIA GPU, install `torch` with CUDA (see `requirements.txt`).

---

### 2 — Create Virtual Environment (recommended)

**Windows (PowerShell / Command Prompt):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ First install takes ~3–5 minutes (PyTorch + Ultralytics are large).  
> 🌐 YOLOv8 model weights (~6 MB for `yolov8n.pt`) are auto-downloaded on **first run**.

---

### 4 — Run the App

```bash
streamlit run app.py
```

The browser opens automatically at **http://localhost:8501**

---

## 🖥️ How to Use

### Live Webcam Mode

1. Select **"📹 Live Webcam"** in the sidebar.
2. Click the **START** button in the video panel.
3. Allow browser camera access when prompted.
4. Objects are detected in real time with bounding boxes, labels, and confidence scores.
5. Click **"📷 Save Screenshot"** to capture the current frame.

### Image Upload Mode

1. Select **"🖼️ Image Upload"** in the sidebar.
2. Drag & drop any `.jpg / .png / .webp` image.
3. The annotated result appears on the right with a download button.

### Sidebar Controls

| Control | Description |
|---------|-------------|
| **Mode** | Switch between live webcam and static image detection |
| **Confidence Threshold** | Slider (10%–95%): filters out weak detections |
| **YOLOv8 Model** | `n` = fastest · `s` = balanced · `m` = most accurate |
| **Download Logs** | Export all session detections as CSV |
| **Session Stats** | Live count of total detections + screenshots |

---

## 🔧 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Layer                          │
│            Webcam (WebRTC)  /  Image Upload                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Preprocessing                            │
│        Frame decode · BGR conversion · model resize         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                     Model Layer                             │
│          YOLOv8n/s/m  (pretrained on COCO 80 classes)       │
│           ultralytics YOLO(frame, verbose=False)            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   Postprocessing                            │
│   Extract boxes · class IDs · confidence scores            │
│   Draw bounding boxes · label pills · HUD overlay          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                     Output Layer                            │
│   Annotated video stream (WebRTC)  /  Annotated PNG         │
│   Live detection list · Metrics panel · CSV log             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 How It Works

### Real-Time Detection (Webcam)

1. **streamlit-webrtc** opens a WebRTC peer connection between browser and server.
2. Each incoming frame is passed to `ObjectDetector.recv()` (runs in a background thread).
3. The frame is decoded from WebRTC's AV format into an OpenCV BGR array.
4. YOLOv8 runs inference → returns `boxes`, `class IDs`, and `confidence scores`.
5. `draw_detections()` draws colored bounding boxes + label pills on the frame.
6. `draw_hud()` overlays FPS and object count in the top-left corner.
7. The annotated frame is re-encoded and sent back to the browser.
8. Detection data is stored in a thread-safe `threading.Lock()` shared with Streamlit.
9. `streamlit-autorefresh` polls the shared state every 500 ms to update the stats panel.

### FPS Calculation

FPS is calculated inside the VideoProcessor with a rolling 1-second window:
```python
self._frame_count += 1
elapsed = time.time() - self._fps_timer
if elapsed >= 1.0:
    fps = self._frame_count / elapsed
    self._frame_count = 0
    self._fps_timer = time.time()
```

### Thread Safety

The `ObjectDetector` uses a `threading.Lock()` to protect shared state:
```python
with self.lock:
    self.detections   = detections
    self.fps          = fps
    self.latest_frame = annotated.copy()
```
The Streamlit UI thread always acquires the same lock before reading:
```python
with ctx.video_processor.lock:
    dets = list(ctx.video_processor.detections)
```

---

## 📦 COCO Object Classes (80 total)

person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, TV, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera not working | Check browser camera permissions · Try Chrome/Edge |
| `streamlit-webrtc` install error | Upgrade pip: `pip install --upgrade pip` |
| Model download fails | Check internet connection; model is ~6 MB |
| Low FPS on CPU | Use `yolov8n.pt` (fastest) · Close other heavy apps |
| Port 8501 busy | Run: `streamlit run app.py --server.port 8502` |
| Screenshots not saving | Make sure `screenshots/` folder exists (auto-created) |

---

## 🚀 Performance Tips

- **CPU (i5/i7):** Use `yolov8n.pt` → expect 8–15 FPS
- **GPU (NVIDIA):** Use `yolov8s.pt` → expect 25–40 FPS  
- **Confidence threshold 0.5** is a good default; lower it to catch more objects
- Frame resolution is determined by your webcam; lower resolution = higher FPS

---

## 📊 Detection Logs

Every detection is appended to `logs/detection_logs.csv`:

```csv
timestamp,object_name,confidence_%,x1,y1,x2,y2
2025-01-15 14:30:22,person,87.4,120,45,380,490
2025-01-15 14:30:22,cell phone,73.1,200,300,280,420
```

Download the full CSV from the sidebar at any time.

---

*Built with Python · Ultralytics YOLOv8 · Streamlit · OpenCV*
