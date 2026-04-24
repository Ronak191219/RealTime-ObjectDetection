# 🎯 Real-Time Object Detection System

A real-time object detection and recognition system built with **YOLOv8**, **OpenCV**, and **Streamlit**. Detects 80 COCO object classes from live webcam feed or uploaded images with a clean dark-themed UI.

---

## 📸 Demo

| Image Upload Mode | Live Detection |
|---|---|
| Upload any image → instant detection | Press START → live webcam feed with bounding boxes |

---

## ✨ Features

- 🎥 **Live Webcam Detection** — real-time bounding boxes + labels
- 🖼️ **Image Upload Detection** — detect objects in any image file
- 📊 **Live Stats Panel** — FPS, object count, confidence scores
- 🎚️ **Confidence Threshold Slider** — filter weak detections (10–95%)
- 📷 **Screenshot Capture** — save annotated frames as JPG
- 📥 **Download Annotated Image** — export results as PNG
- 📂 **CSV Detection Log** — every detection timestamped and saved
- ⚡ **3 Model Options** — YOLOv8n (fastest) · YOLOv8s · YOLOv8m

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| UI Framework | Streamlit |
| Object Detection | YOLOv8 (Ultralytics) |
| Video Processing | OpenCV |
| Deep Learning | PyTorch (CPU/GPU) |
| Dataset | COCO 80 Classes |

---

## 📁 Project Structure

```
Real-Time-Object-Detection/
├── app.py                  ← Main application (single file)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── .gitignore              ← Git ignore rules
├── screenshots/            ← Saved frame captures (auto-created)
└── logs/
    └── detection_logs.csv  ← Detection history CSV (auto-created)
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Webcam (for live detection)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Ronak191219/Real-Time-Object-Detection.git
cd Real-Time-Object-Detection

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install streamlit ultralytics opencv-python numpy Pillow

# 4. Run the app
streamlit run app.py
```

> **Note:** YOLOv8 model weights (~6 MB) are auto-downloaded from Ultralytics on first run.

---

## 🎮 How to Use

### Live Webcam Mode
1. Select **📹 Live Webcam** in the sidebar
2. Click **▶ START** button
3. Objects detected in real-time with bounding boxes
4. Click **📷 Screenshot** to save a frame
5. Click **⏹ STOP** to end the session

### Image Upload Mode
1. Select **🖼️ Image Upload** in the sidebar
2. Drag & drop or click to upload any image
3. Detection results appear instantly on the right
4. Click **📥 Download Annotated Image** to save

---

## 🧠 Architecture

```
Input (Webcam / Image)
        ↓
Preprocessing (BGR conversion, resize)
        ↓
YOLOv8 Inference (ultralytics)
        ↓
Postprocessing (boxes, labels, confidence)
        ↓
draw_detections() → annotated frame
        ↓
Streamlit UI (st.empty().image())
```

### Threading Model (Webcam)
```
Background Thread          Streamlit UI Thread
─────────────────          ───────────────────
cv2.VideoCapture(0)   →    while cam_running:
YOLOv8 inference      →        cam.read()        ← threading.Lock
draw bounding boxes   →        st.image(frame)
store under Lock      →        update stats
```

---

## 📦 COCO 80 Object Classes

`person` `bicycle` `car` `motorcycle` `airplane` `bus` `train` `truck` `boat` `traffic light` `fire hydrant` `stop sign` `bottle` `wine glass` `cup` `fork` `knife` `spoon` `bowl` `banana` `apple` `sandwich` `orange` `broccoli` `carrot` `hot dog` `pizza` `donut` `cake` `chair` `couch` `potted plant` `bed` `dining table` `toilet` `TV` `laptop` `mouse` `remote` `keyboard` `cell phone` `microwave` `oven` `toaster` `sink` `refrigerator` `book` `clock` `vase` `scissors` `teddy bear` `hair drier` `toothbrush` *(and more)*

---

## 📊 Model Comparison

| Model | Size | Speed (CPU) | Accuracy |
|-------|------|-------------|----------|
| YOLOv8n | ~6 MB | Fastest (~8-15 FPS) | Good |
| YOLOv8s | ~22 MB | Balanced (~5-10 FPS) | Better |
| YOLOv8m | ~52 MB | Slower (~2-5 FPS) | Best |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Webcam not opening | Close Teams/Zoom/Discord; check Device Manager |
| Low FPS | Use `yolov8n.pt`; close heavy background apps |
| Port 8501 busy | Run `streamlit run app.py --server.port 8502` |
| Model not downloading | Check internet connection |
| `pip install` encoding error | Run: `pip install streamlit ultralytics opencv-python numpy Pillow` directly |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋 Author

Built as an AI/ML lab project using Python · YOLOv8 · OpenCV · Streamlit