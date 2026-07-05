# 🎯 Real-Time Object Detection System

A modern **Real-Time Object Detection System** built using **YOLOv8**, **OpenCV**, and **Streamlit**. The application detects **80 COCO object classes** from uploaded images and supports **live webcam detection** when running locally. It features an interactive dashboard, real-time statistics, screenshot capture, detection logging, and downloadable annotated images.

---

# 🚀 Live Demo

🔗 **https://realtime-objectdetection-ronak.streamlit.app/**

> **Note**
>
> Due to Streamlit Cloud limitations, **Live Webcam Detection is disabled** in the deployed application because cloud servers cannot access your local webcam.
>
> ✅ Image Upload Detection works perfectly online.
>
> For Live Webcam Detection run locally:
>
> ```bash
> streamlit run app.py
> ```

---

# ✨ Features

- 🎥 **Live Webcam Detection** — real-time bounding boxes + labels 
- 🖼️ **Image Upload Detection** — detect objects in any image file 
- 📊 **Live Stats Panel** — FPS, object count, confidence scores
- 🎚️ **Confidence Threshold Slider** — filter weak detections (10–95%) 
- 📷 **Screenshot Capture** — save annotated frames as JPG 
- 📥 **Download Annotated Image** — export results as PNG 
- 📂 **CSV Detection Log** — every detection timestamped and saved 
- ⚡ **3 Model Options** — YOLOv8n (fastest) · YOLOv8s · YOLOv8m

---

# 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Deep Learning | YOLOv8 (Ultralytics) |
| Computer Vision | OpenCV |
| Frontend | Streamlit |
| Image Processing | Pillow |
| Numerical Computing | NumPy |
| Model Backend | PyTorch |
| Dataset | COCO 80 Classes |

---

#  Project Structure

Real-Time-Object-Detection/
├── app.py                  ← Main application (single file)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── .gitignore              ← Git ignore rules
├── screenshots/            ← Saved frame captures (auto-created)
└── logs/
    └── detection_logs.csv  ← Detection history CSV (auto-created)

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/Ronak191219/Real-Time-Object-Detection.git

cd Real-Time-Object-Detection
```

Create virtual environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
streamlit run app.py
```

---

# 🎮 Usage

## 📹 Live Webcam Detection

- Select **Live Webcam**
- Click **START**
- Detect objects in real time
- Save screenshots
- View live FPS and object count

---

## 🖼 Image Upload Detection

- Upload an image
- Run YOLOv8 inference
- View detected objects
- Download annotated image
- Detection automatically logged

---

# 🧠 Detection Pipeline

```text
Input Image / Webcam
        │
        ▼
Preprocessing
        │
        ▼
YOLOv8 Inference
        │
        ▼
Bounding Boxes
        │
        ▼
Confidence Filtering
        │
        ▼
Visualization
        │
        ▼
Streamlit Dashboard
```

---

# 📊 Supported YOLO Models

| Model | Speed | Accuracy |
|--------|--------|----------|
| YOLOv8n | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| YOLOv8s | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| YOLOv8m | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 📊 Model Comparison

| Model | Size | Speed (CPU) | Accuracy | 
|-------|------|-------------|----------| 
| YOLOv8n | ~6 MB | Fastest (~8-15 FPS) | Good | 
| YOLOv8s | ~22 MB | Balanced (~5-10 FPS) | Better | 
| YOLOv8m | ~52 MB | Slower (~2-5 FPS) | Best | 

---

# 📦 Supported COCO Classes

The application supports all **80 COCO classes**, including:

- Person
- Bicycle
- Car
- Motorcycle
- Bus
- Truck
- Airplane
- Dog
- Cat
- Bird
- Chair
- Laptop
- TV
- Cell Phone
- Bottle
- Book

...and many more.

---

# ☁ Deployment

| Platform | Status |
|----------|--------|
| Streamlit Cloud | ✅ Image Upload |
| Local Machine | ✅ Webcam + Image Upload |
| Webcam on Cloud | ❌ Not Supported |

---

#  Troubleshooting | Problem | Solution | 

|---------|----------| 
| Webcam not opening | Close Teams/Zoom/Discord; check Device Manager | 
| Low FPS | Use yolov8n.pt; close heavy background apps | 
| Port 8501 busy | Run streamlit run app.py --server.port 8502 | 
| Model not downloading | Check internet connection | 
| pip install encoding error | Run: pip install streamlit ultralytics opencv-python numpy Pillow directly | 

---

# 🚀 Future Improvements

- Video File Detection
- Object Tracking
- Object Counting
- Custom YOLO Model Upload
- Detection Analytics Dashboard
- PDF Detection Report
- Cloud Webcam Support using streamlit-webrtc

---

# 👨‍💻 Author

**Ronak**

B.Tech – Computer Science & Engineering (AI & ML)

### GitHub

https://github.com/Ronak191219

---

# ⭐ Like this Project?

If you found this project useful,

⭐ **Please give this repository a Star.**

It helps others discover the project and motivates future improvements.