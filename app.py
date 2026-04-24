"""
╔══════════════════════════════════════════════════════════════════╗
║        Real-Time Object Detection & Recognition System          ║
║        YOLOv8 · OpenCV · Streamlit                              ║
║        Works on Windows/Mac/Linux — No 'av' module needed       ║
╚══════════════════════════════════════════════════════════════════╝

ARCHITECTURE:
  ┌──────────────────────────────────────────────────────────────┐
  │  Background Thread  (CameraStream)                           │
  │   cv2.VideoCapture(0) → YOLOv8 Inference                    │
  │   → Annotated Frame stored under threading.Lock             │
  └───────────────────────────┬──────────────────────────────────┘
                              │  threading.Lock (thread-safe)
  ┌───────────────────────────▼──────────────────────────────────┐
  │  Streamlit Main Thread                                       │
  │   while cam_running:                                         │
  │       frame = cam.read()  → st.empty().image(frame)          │
  │       update stats panel  → time.sleep(0.08)                 │
  └──────────────────────────────────────────────────────────────┘
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os
import io
import csv
import time
import threading
from datetime import datetime

# ─── Third-Party Libraries ────────────────────────────────────────────────────
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Real-Time Object Detection System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Dark Sci-Fi Theme
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;500;600&display=swap');

:root {
  --cyan:   #00d4ff;
  --purple: #7b61ff;
  --red:    #ff6b6b;
  --green:  #00ff88;
  --yellow: #ffd700;
  --bg:     #060e1a;
  --bg2:    #0a1628;
  --card:   rgba(0,212,255,0.04);
  --border: rgba(0,212,255,0.15);
}

/* ── Global ── */
.stApp {
  background: var(--bg) !important;
}
*, body, p, div, span, label {
  font-family: 'Exo 2', sans-serif !important;
  color: #c8d8e8;
}
#MainMenu, footer { visibility: hidden; }

/* ── Sidebar collapse/expand arrow button ── */
[data-testid="collapsedControl"] {
  display: flex !important;
  visibility: visible !important;
  background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(123,97,255,0.12)) !important;
  border: 1px solid rgba(0,212,255,0.4) !important;
  border-radius: 0 10px 10px 0 !important;
  width: 28px !important;
  height: 52px !important;
  top: 50% !important;
  transform: translateY(-50%) !important;
  box-shadow: 2px 0 16px rgba(0,212,255,0.15) !important;
  transition: all 0.25s !important;
  align-items: center !important;
  justify-content: center !important;
}
[data-testid="collapsedControl"]:hover {
  background: linear-gradient(135deg, rgba(0,212,255,0.28), rgba(123,97,255,0.28)) !important;
  box-shadow: 2px 0 24px rgba(0,212,255,0.3) !important;
  width: 32px !important;
}
[data-testid="collapsedControl"] svg {
  fill: #00d4ff !important;
  width: 16px !important;
  height: 16px !important;
}
/* Arrow inside open sidebar (collapse arrow) */
[data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"] {
  background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(123,97,255,0.08)) !important;
  border: 1px solid rgba(0,212,255,0.3) !important;
  border-radius: 8px !important;
  color: #00d4ff !important;
  transition: all 0.25s !important;
}
[data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"]:hover {
  background: rgba(0,212,255,0.2) !important;
  box-shadow: 0 0 14px rgba(0,212,255,0.25) !important;
}

/* ── Top Title ── */
.main-title {
  font-family: 'Orbitron', monospace !important;
  font-size: clamp(1.2rem, 2.8vw, 2rem);
  font-weight: 900;
  background: linear-gradient(90deg, var(--cyan) 0%, var(--purple) 50%, var(--red) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-align: center;
  letter-spacing: 4px;
  margin: 0.4rem 0 0.1rem;
  text-transform: uppercase;
}
.subtitle {
  text-align: center;
  color: #2a4060 !important;
  font-size: 0.72rem;
  letter-spacing: 5px;
  text-transform: uppercase;
  margin-bottom: 1.4rem;
}

/* ── Section Headers ── */
.sec-head {
  font-family: 'Orbitron', monospace !important;
  font-size: 0.68rem !important;
  color: var(--cyan) !important;
  letter-spacing: 3px;
  text-transform: uppercase;
  border-bottom: 1px solid var(--border);
  padding-bottom: 6px;
  margin: 1rem 0 0.7rem;
  display: flex;
  align-items: center;
  gap: 6px;
}

/* ── Metric Cards ── */
.metric-row {
  display: flex;
  gap: 10px;
  margin: 0.7rem 0;
}
.metric-card {
  flex: 1;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.8rem 0.4rem 0.6rem;
  text-align: center;
  transition: border-color 0.25s, box-shadow 0.25s;
}
.metric-card:hover {
  border-color: var(--cyan);
  box-shadow: 0 0 16px rgba(0,212,255,0.12);
}
.metric-val {
  font-family: 'Orbitron', monospace !important;
  font-size: 1.8rem;
  font-weight: 700;
  color: var(--cyan) !important;
  line-height: 1;
}
.metric-lbl {
  font-size: 0.6rem;
  color: #2a4060 !important;
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-top: 5px;
}

/* ── Detection Items ── */
.det-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: rgba(0,212,255,0.035);
  border-left: 3px solid var(--cyan);
  border-radius: 0 8px 8px 0;
  padding: 7px 14px;
  margin: 5px 0;
  font-size: 0.87rem;
  transition: background 0.2s;
}
.det-item:hover { background: rgba(0,212,255,0.07); }
.det-name { color: #c8d8e8 !important; font-weight: 500; }
.det-conf {
  color: var(--cyan) !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 0.75rem;
  font-weight: 700;
}

/* ── Status Badges ── */
.badge-live {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(0,255,136,0.06);
  border: 1px solid rgba(0,255,136,0.35);
  color: var(--green) !important;
  border-radius: 20px; padding: 3px 14px;
  font-size: 0.78rem; font-weight: 600;
  animation: pulse 2s infinite;
}
.badge-off {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(255,107,107,0.06);
  border: 1px solid rgba(255,107,107,0.3);
  color: var(--red) !important;
  border-radius: 20px; padding: 3px 14px;
  font-size: 0.78rem;
}
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(0,255,136,0.25); }
  50%       { box-shadow: 0 0 0 6px rgba(0,255,136,0); }
}

/* ── Info / Empty Box ── */
.info-box {
  background: rgba(0,212,255,0.03);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 13px 15px;
  font-size: 0.82rem;
  color: #3a5a70 !important;
  line-height: 1.65;
}
.video-placeholder {
  background: rgba(0,212,255,0.02);
  border: 1px dashed rgba(0,212,255,0.15);
  border-radius: 12px;
  padding: 5rem 2rem;
  text-align: center;
  color: #2a4060 !important;
  font-size: 1rem;
  letter-spacing: 2px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #04090f !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: #a0b4c4 !important; }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, rgba(0,212,255,0.06), rgba(123,97,255,0.06)) !important;
  border: 1px solid rgba(0,212,255,0.3) !important;
  color: var(--cyan) !important;
  font-family: 'Exo 2', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: 1.5px !important;
  border-radius: 8px !important;
  width: 100% !important;
  padding: 0.45rem 0.8rem !important;
  transition: all 0.2s !important;
  text-transform: uppercase !important;
  font-size: 0.78rem !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(0,212,255,0.18), rgba(123,97,255,0.18)) !important;
  border-color: var(--cyan) !important;
  box-shadow: 0 0 20px rgba(0,212,255,0.2) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:disabled {
  opacity: 0.3 !important;
  cursor: not-allowed !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
  background: var(--cyan) !important;
}

/* ── Image ── */
[data-testid="stImage"] img {
  border-radius: 10px !important;
  border: 1px solid var(--border) !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
  background: linear-gradient(135deg, rgba(0,212,255,0.06), rgba(123,97,255,0.06)) !important;
  border: 1px solid rgba(0,212,255,0.3) !important;
  color: var(--cyan) !important;
  border-radius: 8px !important;
  font-family: 'Exo 2', sans-serif !important;
  font-size: 0.82rem !important;
  width: 100% !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: rgba(0,212,255,0.02) !important;
  border: 1px dashed rgba(0,212,255,0.2) !important;
  border-radius: 10px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.2); border-radius: 4px; }

/* ── Divider ── */
hr { border-color: var(--border) !important; opacity: 1 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & FOLDER SETUP
# ══════════════════════════════════════════════════════════════════════════════

SCREENSHOTS_DIR = "screenshots"
LOGS_DIR        = "logs"
LOG_FILE        = os.path.join(LOGS_DIR, "detection_logs.csv")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,        exist_ok=True)

# Fixed consistent color per COCO class (same color = same object always)
np.random.seed(42)
CLASS_COLORS = np.random.randint(60, 220, size=(80, 3), dtype=np.uint8).tolist()


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING  (cached — loads once, survives Streamlit reruns)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(name: str = "yolov8n.pt") -> YOLO:
    """
    Load YOLOv8 weights. Auto-downloads from Ultralytics on first run (~6 MB).
    @st.cache_resource ensures the model loads only ONCE per session.
    """
    return YOLO(name)


# ══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND CAMERA STREAM  (Thread-based, no 'av', no WebRTC)
# ══════════════════════════════════════════════════════════════════════════════

class CameraStream:
    """
    Thin wrapper so session_state can hold a 'running' flag.
    Actual capture happens in the Streamlit main thread to avoid
    Windows threading issues with cv2.VideoCapture.
    """
    def __init__(self):
        self.running = False
        self.cap     = None

    def open(self):
        """Open webcam in the MAIN thread — required on Windows."""
        for idx in range(3):
            c = cv2.VideoCapture(idx)
            if c.isOpened():
                ok, frm = c.read()
                if ok and frm is not None:
                    c.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    c.set(cv2.CAP_PROP_BUFFERSIZE,   1)
                    self.cap     = c
                    self.running = True
                    return True
            c.release()
        return False

    def read(self):
        if self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            return (frame if ok else None)
        return None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_detections(frame: np.ndarray, results, conf_thresh: float):
    """
    Parse YOLO results and draw bounding boxes + label pills on the frame.

    Steps:
      - Skip boxes below conf_thresh
      - Draw filled rectangle (bounding box)
      - Draw label pill with class name + confidence %
      - Collect detection data into list of dicts

    Returns:
        annotated_frame (np.ndarray),  detections (list of dict)
    """
    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            # ── Confidence filter ──────────────────────────────────────────
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue

            # ── Extract class info ─────────────────────────────────────────
            cls_id   = int(box.cls[0])
            cls_name = result.names[cls_id]

            # ── Coordinates ────────────────────────────────────────────────
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ── Color for this class ───────────────────────────────────────
            color = tuple(CLASS_COLORS[cls_id % 80])

            # ── Bounding Box ───────────────────────────────────────────────
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ── Label Pill ─────────────────────────────────────────────────
            label = f"{cls_name}  {conf:.0%}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            ly = max(y1, th + 12)

            # Pill background
            cv2.rectangle(
                frame,
                (x1, ly - th - 10),
                (x1 + tw + 10, ly + bl - 2),
                color, cv2.FILLED
            )
            # Pill text (white)
            cv2.putText(
                frame, label, (x1 + 5, ly - 4),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
            )

            # ── Store result ───────────────────────────────────────────────
            detections.append({
                "name":       cls_name,
                "confidence": round(conf * 100, 2),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    return frame, detections


def draw_hud(frame: np.ndarray, fps: float, n: int) -> np.ndarray:
    """
    Draw semi-transparent HUD with FPS and object count in top-left corner.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (210, 72), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FPS : {fps:5.1f}",
                (10, 26), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 230, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, f"OBJS: {n:3d}",
                (10, 56), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 200, 255), 2, cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
#  CSV LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def init_log():
    """Create CSV with headers if it doesn't exist yet."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "object_name", "confidence_%",
                 "x1", "y1", "x2", "y2"]
            )


def log_detections(detections: list):
    """Append detections to CSV log with timestamp."""
    if not detections:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        for d in detections:
            w.writerow([ts, d["name"], d["confidence"],
                        d["x1"], d["y1"], d["x2"], d["y2"]])


# ══════════════════════════════════════════════════════════════════════════════
#  HTML COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def html_metric_cards(fps: float, n: int, total: int) -> str:
    """Build the 3-column metric cards HTML."""
    return f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-val">{fps:.1f}</div>
        <div class="metric-lbl">FPS</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">{n}</div>
        <div class="metric-lbl">Objects</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">{total}</div>
        <div class="metric-lbl">Total Logged</div>
      </div>
    </div>"""


def html_detection_list(dets: list) -> str:
    """Build the detection list HTML with name + confidence."""
    if not dets:
        return '<div class="info-box">No objects detected above threshold.</div>'
    return "".join(
        f'<div class="det-item">'
        f'<span class="det-name">🏷️ &nbsp;{d["name"]}</span>'
        f'<span class="det-conf">{d["confidence"]:.1f}%</span>'
        f'</div>'
        for d in dets
    )


def html_image_stats(n: int, ms: float, conf: float) -> str:
    """Build the 3-column stats row for image upload mode."""
    return f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-val">{n}</div>
        <div class="metric-lbl">Objects</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">{ms:.0f}</div>
        <div class="metric-lbl">MS</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">{conf:.0%}</div>
        <div class="metric-lbl">Min Conf</div>
      </div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialise all session variables once per browser session."""
    defaults = {
        "cam_running":       False,
        "camera_stream":     None,
        "total_detections":  0,
        "screenshots_taken": 0,
        "last_frame":        None,   # latest BGR frame for screenshot
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """
    Build the sidebar controls.
    Returns: (mode, confidence_threshold, model_name)
    """
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="text-align:center; padding: 0.5rem 0 0.8rem;">
          <span style="font-family:Orbitron,monospace; font-size:1.05rem;
                       font-weight:700; color:#00d4ff; letter-spacing:2px;">
            🎯 DETECTION SYSTEM
          </span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Mode ─────────────────────────────────────────────────────────────
        st.markdown('<div class="sec-head">⚡ MODE</div>', unsafe_allow_html=True)
        mode = st.radio(
            "Select Mode",
            ["📹 Live Webcam", "🖼️ Image Upload"],
            label_visibility="collapsed",
        )

        # ── Parameters ───────────────────────────────────────────────────────
        st.markdown('<div class="sec-head">🎚 PARAMETERS</div>', unsafe_allow_html=True)

        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.10, max_value=0.95,
            value=0.50, step=0.05,
            help="Only detections above this score are shown."
        )

        model_name = st.selectbox(
            "YOLOv8 Model",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            index=0,
            help="**n** = Nano (fastest)\n**s** = Small (balanced)\n**m** = Medium (most accurate)"
        )

        st.divider()

        # ── Session Stats ─────────────────────────────────────────────────────
        st.markdown('<div class="sec-head">📈 SESSION</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Detected",    st.session_state.total_detections)
        c2.metric("Screenshots", st.session_state.screenshots_taken)

        st.divider()

        # ── Log Download ──────────────────────────────────────────────────────
        st.markdown('<div class="sec-head">📂 LOGS</div>', unsafe_allow_html=True)
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE) as f:
                st.download_button(
                    "📥 detection_logs.csv",
                    data      = f.read(),
                    file_name = "detection_logs.csv",
                    mime      = "text/csv"
                )
        else:
            st.caption("No logs yet. Run a detection first.")

        st.divider()

        # ── Info ──────────────────────────────────────────────────────────────
        st.markdown("""
        <div class="info-box">
          Detects <strong style="color:#00d4ff">80 COCO classes</strong> —
          people, vehicles, animals, everyday objects &amp; more.<br><br>
          Model weights auto-download from Ultralytics on first run (~6 MB).
        </div>
        """, unsafe_allow_html=True)

    return mode, confidence, model_name


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE WEBCAM SECTION
# ══════════════════════════════════════════════════════════════════════════════

def section_live_webcam(confidence: float, model_name: str):
    """
    Live webcam detection — cv2.VideoCapture runs in the MAIN thread.
    This avoids the Windows threading issue where VideoCapture silently
    fails when opened from a background/daemon thread.
    """
    model = load_model(model_name)

    col_video, col_info = st.columns([2.2, 1], gap="large")

    with col_video:
        st.markdown('<div class="sec-head">📡 LIVE CAMERA FEED</div>',
                    unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        start_btn = b1.button("▶  START",
                              disabled=st.session_state.cam_running,
                              key="btn_start")
        stop_btn  = b2.button("⏹  STOP",
                              disabled=not st.session_state.cam_running,
                              key="btn_stop")
        snap_btn  = b3.button("📷  Screenshot",
                              disabled=not st.session_state.cam_running,
                              key="btn_snap")
        video_ph = st.empty()

    with col_info:
        st.markdown('<div class="sec-head">📊 LIVE STATS</div>',
                    unsafe_allow_html=True)
        status_ph  = st.empty()
        metrics_ph = st.empty()
        dets_hdr   = st.empty()
        dets_ph    = st.empty()

    # ── START ───────────────────────────────────────────────────────────────
    if start_btn and not st.session_state.cam_running:
        cam = CameraStream()
        if cam.open():                          # opens in main thread ✅
            st.session_state.camera_stream = cam
            st.session_state.cam_running   = True
        else:
            st.error("❌ Cannot open webcam. Make sure no other app is using it.")
            return

    # ── STOP ────────────────────────────────────────────────────────────────
    if stop_btn and st.session_state.cam_running:
        if st.session_state.camera_stream:
            st.session_state.camera_stream.stop()
            st.session_state.camera_stream = None
        st.session_state.cam_running = False

    # ── SCREENSHOT ──────────────────────────────────────────────────────────
    if snap_btn and st.session_state.last_frame is not None:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SCREENSHOTS_DIR, f"detection_{ts}.jpg")
        cv2.imwrite(path, st.session_state.last_frame)
        st.session_state.screenshots_taken += 1
        st.toast(f"📸  Saved → {path}", icon="✅")

    # ── LIVE CAPTURE LOOP (main thread) ─────────────────────────────────────
    if st.session_state.cam_running and st.session_state.camera_stream:
        cam = st.session_state.camera_stream
        cam.conf_threshold = confidence

        status_ph.markdown('<span class="badge-live">● &nbsp;LIVE</span>',
                           unsafe_allow_html=True)
        dets_hdr.markdown('<div class="sec-head">🔍 DETECTED OBJECTS</div>',
                          unsafe_allow_html=True)

        frame_count = 0
        fps_timer   = time.time()
        current_fps = 0.0

        while st.session_state.cam_running:
            frame = cam.read()

            if frame is None:
                video_ph.markdown(
                    '<div class="info-box">⚠️ Lost camera feed. Press STOP then START.</div>',
                    unsafe_allow_html=True)
                break

            # ── Inference ────────────────────────────────────────────────
            results = model(frame, verbose=False)
            annotated, dets = draw_detections(frame, results, confidence)

            # ── FPS ──────────────────────────────────────────────────────
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                frame_count = 0
                fps_timer   = time.time()

            # ── HUD ──────────────────────────────────────────────────────
            annotated = draw_hud(annotated, current_fps, len(dets))

            # ── Display ──────────────────────────────────────────────────
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_ph.image(rgb, channels="RGB", use_container_width=True)

            st.session_state.last_frame        = annotated
            st.session_state.total_detections += len(dets)
            log_detections(dets)

            metrics_ph.markdown(
                html_metric_cards(current_fps, len(dets), st.session_state.total_detections),
                unsafe_allow_html=True)
            dets_ph.markdown(html_detection_list(dets), unsafe_allow_html=True)

            time.sleep(0.04)   # ~25 fps UI cap

    else:
        status_ph.markdown('<span class="badge-off">○ &nbsp;CAMERA OFF</span>',
                           unsafe_allow_html=True)
        metrics_ph.markdown(
            html_metric_cards(0.0, 0, st.session_state.total_detections),
            unsafe_allow_html=True)
        dets_hdr.markdown('<div class="sec-head">🔍 DETECTED OBJECTS</div>',
                          unsafe_allow_html=True)
        dets_ph.markdown(
            '<div class="info-box">Press <strong>▶ START</strong> to activate the camera.</div>',
            unsafe_allow_html=True)
        video_ph.markdown(
            '<div class="video-placeholder">🎥<br><br>Camera feed will appear here</div>',
            unsafe_allow_html=True)

def section_image_upload(confidence: float, model_name: str):
    """
    Static image detection.

    Layout:
      Left : Original uploaded image
      Right: Annotated result + metrics + detection list + download button

    Steps:
      1. User uploads image via st.file_uploader
      2. PIL Image → OpenCV BGR array
      3. YOLOv8 inference (timed)
      4. draw_detections() → annotated BGR → RGB for display
      5. Results shown with download option
    """
    model = load_model(model_name)

    col_orig, col_result = st.columns(2, gap="large")

    # ── Left: Upload + Original ───────────────────────────────────────────────
    with col_orig:
        st.markdown('<div class="sec-head">📤 UPLOAD IMAGE</div>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag & drop or click to upload",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Supports JPG, JPEG, PNG, BMP, WebP",
        )
        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Original", use_container_width=True)

    # ── Right: Detection Result ───────────────────────────────────────────────
    with col_result:
        st.markdown('<div class="sec-head">🎯 DETECTED OBJECTS</div>',
                    unsafe_allow_html=True)

        if not uploaded:
            st.markdown(
                '<div class="info-box">📁 Upload an image on the left to run YOLOv8 detection.</div>',
                unsafe_allow_html=True,
            )
            return

        # ── Inference ─────────────────────────────────────────────────────────
        with st.spinner("Running YOLOv8 inference …"):
            # Convert PIL RGB → OpenCV BGR
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Time the inference
            t0      = time.perf_counter()
            results = model(img_bgr, verbose=False)
            ms      = (time.perf_counter() - t0) * 1000

            # Draw bounding boxes + labels
            ann_bgr, detections = draw_detections(img_bgr, results, confidence)

            # Convert back to RGB for Streamlit
            ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

        # ── Show annotated result ──────────────────────────────────────────────
        st.image(ann_rgb, caption="Detected Objects", use_container_width=True)

        # ── Stats row ─────────────────────────────────────────────────────────
        st.markdown(html_image_stats(len(detections), ms, confidence),
                    unsafe_allow_html=True)

        # ── Detection list ────────────────────────────────────────────────────
        st.markdown(html_detection_list(detections), unsafe_allow_html=True)

        # ── Download annotated image ──────────────────────────────────────────
        if detections:
            buf = io.BytesIO()
            Image.fromarray(ann_rgb).save(buf, format="PNG")
            st.download_button(
                label     = "📥  Download Annotated Image",
                data      = buf.getvalue(),
                file_name = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime      = "image/png",
            )

        # ── Log results ────────────────────────────────────────────────────────
        log_detections(detections)
        st.session_state.total_detections += len(detections)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN  — Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Wire everything together."""

    # ── One-time setup ─────────────────────────────────────────────────────────
    init_session_state()
    init_log()

    # ── Page header ────────────────────────────────────────────────────────────
    st.markdown(
        '<h1 class="main-title">🎯 Real-Time Object Detection System</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">YOLOv8 &nbsp;·&nbsp; OpenCV &nbsp;·&nbsp; '
        'Streamlit &nbsp;·&nbsp; COCO 80 Classes</p>',
        unsafe_allow_html=True,
    )

    # ── Sidebar returns user selections ────────────────────────────────────────
    mode, confidence, model_name = render_sidebar()

    # ── Route to correct section ───────────────────────────────────────────────
    if mode == "📹 Live Webcam":
        section_live_webcam(confidence, model_name)
    else:
        section_image_upload(confidence, model_name)


if __name__ == "__main__":
    main()