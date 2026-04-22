"""
╔══════════════════════════════════════════════════════════════════╗
║       Real-Time Object Detection & Recognition System            ║
║       Stack : YOLOv8 · OpenCV · Streamlit (pure, no WebRTC)      ║
║       Works : Windows / Mac / Linux  — NO 'av' module needed     ║
╚══════════════════════════════════════════════════════════════════╝

Architecture (no WebRTC / no 'av' dependency):
  ┌─────────────────────────────────────────────────────┐
  │  Background Daemon Thread  (CameraStream._loop)     │
  │   cv2.VideoCapture(0)  →  YOLOv8 inference          │
  │   →  annotated frame + detections stored            │
  └────────────────────┬────────────────────────────────┘
                       │  threading.Lock  (thread-safe reads)
  ┌────────────────────▼────────────────────────────────┐
  │  Streamlit Main Thread                              │
  │   while cam_running:                                │
  │       frame = cam.read()       # latest BGR frame   │
  │       video_ph.image(frame)    # st.empty() slot    │
  │       metrics_ph / dets_ph     # stats widgets      │
  │       time.sleep(0.08)         # ~12 fps UI update  │
  └─────────────────────────────────────────────────────┘
"""

# ── Standard Library ───────────────────────────────────────────────────────
import os
import io
import csv
import time
import threading
from datetime import datetime

# ── Third-Party (all installable on Windows without issues) ────────────────
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from webrtc_webcam import section_webcam_webrtc


# ══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title = "Real-Time Object Detection System",
    page_icon  = "🎯",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# ══════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS  — dark sci-fi theme
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;500;600&display=swap');

:root {
  --cyan:   #00d4ff;
  --purple: #7b61ff;
  --red:    #ff6b6b;
  --green:  #00ff88;
  --bg:     #08101c;
  --bg2:    #0d1b2a;
  --card:   rgba(255,255,255,0.03);
  --border: rgba(0,212,255,0.18);
}
.stApp { background: linear-gradient(135deg, var(--bg) 0%, var(--bg2) 60%, var(--bg) 100%); }
*, body, p, div { color: #d0dce8; font-family: 'Exo 2', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.main-title {
  font-family:'Orbitron',monospace; font-size:clamp(1.3rem,3vw,2.1rem); font-weight:900;
  background:linear-gradient(90deg,var(--cyan) 0%,var(--purple) 50%,var(--red) 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  text-align:center; letter-spacing:3px; padding:0.5rem 0 0.2rem;
}
.subtitle {
  text-align:center; color:#3a5068 !important; font-size:0.75rem;
  letter-spacing:4px; text-transform:uppercase; margin-bottom:1.6rem;
}
.sec-head {
  font-family:'Orbitron',monospace; font-size:0.70rem; color:var(--cyan) !important;
  letter-spacing:3px; text-transform:uppercase; border-bottom:1px solid var(--border);
  padding-bottom:5px; margin:1rem 0 0.7rem;
}
.metric-row { display:flex; gap:10px; margin:0.7rem 0; }
.metric-card {
  flex:1; background:var(--card); border:1px solid var(--border);
  border-radius:10px; padding:0.75rem 0.4rem; text-align:center;
}
.metric-val {
  font-family:'Orbitron',monospace; font-size:1.6rem; font-weight:700;
  color:var(--cyan) !important; line-height:1;
}
.metric-lbl { font-size:0.62rem; color:#3a5068 !important; letter-spacing:2px; text-transform:uppercase; margin-top:4px; }
.det-item {
  display:flex; justify-content:space-between; align-items:center;
  background:rgba(0,212,255,0.04); border-left:3px solid var(--cyan);
  border-radius:0 8px 8px 0; padding:5px 12px; margin:4px 0; font-size:0.83rem;
}
.det-name { color:#c8d8e8 !important; }
.det-conf { color:var(--cyan) !important; font-family:'Orbitron',monospace; font-size:0.76rem; }
.badge-active   { display:inline-block; background:rgba(0,255,136,0.08); border:1px solid rgba(0,255,136,0.4); color:var(--green) !important; border-radius:20px; padding:2px 12px; font-size:0.75rem; }
.badge-inactive { display:inline-block; background:rgba(255,107,107,0.08); border:1px solid rgba(255,107,107,0.4); color:var(--red) !important; border-radius:20px; padding:2px 12px; font-size:0.75rem; }
.info-box {
  background:rgba(0,212,255,0.04); border:1px solid var(--border); border-radius:10px;
  padding:11px 13px; font-size:0.81rem; color:#5a7a90 !important; line-height:1.6;
}
[data-testid="stSidebar"] { background:rgba(8,16,28,0.97) !important; border-right:1px solid var(--border) !important; }
.stButton > button {
  background:linear-gradient(135deg,rgba(0,212,255,0.08),rgba(123,97,255,0.08));
  border:1px solid rgba(0,212,255,0.35); color:var(--cyan) !important;
  font-family:'Exo 2',sans-serif; letter-spacing:1px; border-radius:8px; width:100%; transition:all 0.25s;
}
.stButton > button:hover {
  background:linear-gradient(135deg,rgba(0,212,255,0.22),rgba(123,97,255,0.22));
  border-color:var(--cyan); box-shadow:0 0 18px rgba(0,212,255,0.25);
}
[data-testid="stImage"] img { border-radius:10px; border:1px solid var(--border); }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:rgba(0,212,255,0.25); border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  CONSTANTS & FOLDERS
# ══════════════════════════════════════════════════════════════════════════

SCREENSHOTS_DIR = "screenshots"
LOGS_DIR        = "logs"
LOG_FILE        = os.path.join(LOGS_DIR, "detection_logs.csv")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,        exist_ok=True)

# Fixed color per COCO class for visual consistency
np.random.seed(42)
CLASS_COLORS = np.random.randint(50, 230, size=(80, 3), dtype=np.uint8).tolist()


# ══════════════════════════════════════════════════════════════════════════
#  MODEL  (cached — loads once, survives Streamlit reruns)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(name: str = "yolov8n.pt") -> YOLO:
    """
    Load YOLOv8 weights. First run auto-downloads from Ultralytics hub (~6 MB).
    Subsequent runs use the local cache — instant load.
    """
    return YOLO(name)


# ══════════════════════════════════════════════════════════════════════════
#  BACKGROUND CAMERA + INFERENCE THREAD
# ══════════════════════════════════════════════════════════════════════════

class CameraStream:
    """
    Opens the webcam and runs YOLOv8 inference in a daemon thread.

    Thread-safety model:
      - The daemon thread writes  _frame / _detections / _fps  under  _lock.
      - The Streamlit UI thread reads the same fields via  .read()  under  _lock.
      - No shared data is ever accessed without holding the lock.
    """

    def __init__(self, model: YOLO, conf: float = 0.5) -> None:
        self.model          = model
        self.conf_threshold = conf      # can be updated live from sidebar

        self._lock       = threading.Lock()
        self._frame      = None         # latest annotated BGR frame
        self._detections = []
        self._fps        = 0.0
        self._running    = False
        self._thread     = None
        self._opened     = False
        self._open_error = None

    # ── Public ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the background thread."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and release the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def read(self):
        """
        Return (annotated_bgr_frame, detections_list, fps).
        Returns (None, [], 0.0) if no frame is ready yet.
        """
        with self._lock:
            f = self._frame.copy() if self._frame is not None else None
            d = list(self._detections)
            fps = self._fps
        return f, d, fps

    def is_opened(self) -> bool:
        with self._lock:
            return self._opened

    def open_error(self):
        with self._lock:
            return self._open_error

    # ── Daemon loop ─────────────────────────────────────────────────────────

    def _loop(self) -> None:
        """
        Runs in background thread:
          1. Open cv2.VideoCapture(0)   — uses default webcam
          2. Read frame
          3. Run YOLOv8  →  draw boxes + HUD
          4. Store results under lock
          5. Repeat until stop() is called
        """
        cap = self._open_camera()
        if cap is None:
            with self._lock:
                self._opened = False
                if self._open_error is None:
                    self._open_error = (
                        "Cannot open webcam. Try index 0 in Camera Settings, "
                        "then close any app already using the camera."
                    )
            self._running = False
            return

        # Preferred capture resolution (webcam may override)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        with self._lock:
            self._opened = True
            self._open_error = None

        frame_count = 0
        fps_timer   = time.time()
        current_fps = 0.0

        while self._running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            # ── YOLOv8 inference ──────────────────────────────────────────
            results = self.model(frame, verbose=False)

            # ── Draw bounding boxes + labels ──────────────────────────────
            annotated, detections = _draw_detections(
                frame, results, self.conf_threshold
            )

            # ── FPS (1-second rolling window) ─────────────────────────────
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                frame_count = 0
                fps_timer   = time.time()

            # ── HUD (top-left overlay) ────────────────────────────────────
            annotated = _draw_hud(annotated, current_fps, len(detections))

            # ── Update shared state (thread-safe) ─────────────────────────
            with self._lock:
                self._frame      = annotated
                self._detections = detections
                self._fps        = current_fps

        cap.release()
        with self._lock:
            self._opened = False

    def _open_camera(self):
        """
        Probe common camera indices/backends. On Windows, trying DirectShow
        first is usually more reliable than leaving backend selection implicit.
        """
        attempts = []
        candidates = [(0, cv2.CAP_DSHOW), (0, cv2.CAP_MSMF), (1, cv2.CAP_DSHOW),
                      (1, cv2.CAP_MSMF), (2, cv2.CAP_DSHOW), (2, cv2.CAP_MSMF),
                      (0, cv2.CAP_ANY)]

        for index, backend in candidates:
            try:
                cap = cv2.VideoCapture(index, backend)
            except Exception as exc:
                attempts.append(f"index {index} backend {backend}: {exc}")
                continue

            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return cap
            cap.release()
            attempts.append(f"index {index} backend {backend}: not available")

        with self._lock:
            self._open_error = "Tried camera indices 0, 1 and 2, but none could be opened."
            if attempts:
                self._open_error += " " + " | ".join(attempts[:4])
        return None


# ══════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _draw_detections(frame: np.ndarray, results, conf_thresh: float):
    """
    Draw bounding boxes and label pills onto the frame in-place.

    Returns:
        (annotated_frame,  list_of_dicts)
        Each dict: { name, confidence, x1, y1, x2, y2 }
    """
    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue

            cls_id   = int(box.cls[0])
            cls_name = result.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = tuple(CLASS_COLORS[cls_id % 80])

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label pill
            label = f"{cls_name}  {conf:.0%}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            ly = max(y1, th + 10)
            cv2.rectangle(frame, (x1, ly-th-8), (x1+tw+8, ly+bl-4), color, cv2.FILLED)
            cv2.putText(frame, label, (x1+4, ly-4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

            detections.append({
                "name": cls_name, "confidence": round(conf*100, 2),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    return frame, detections


def _draw_hud(frame: np.ndarray, fps: float, n: int) -> np.ndarray:
    """Overlay FPS and detection count on the top-left corner."""
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (200, 70), (0,0,0), cv2.FILLED)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"FPS : {fps:5.1f}", (10,25),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,230,100), 2, cv2.LINE_AA)
    cv2.putText(frame, f"OBJS: {n:3d}", (10,55),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,200,255), 2, cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════

def _init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp","object_name","confidence_%","x1","y1","x2","y2"]
            )

def _log(detections):
    if not detections:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        for d in detections:
            w.writerow([ts, d["name"], d["confidence"],
                        d["x1"], d["y1"], d["x2"], d["y2"]])


# ══════════════════════════════════════════════════════════════════════════
#  HTML SNIPPETS FOR STATS PANEL
# ══════════════════════════════════════════════════════════════════════════

def _cards_html(fps, n, total):
    return f"""
    <div class="metric-row">
      <div class="metric-card"><div class="metric-val">{fps:.1f}</div>
        <div class="metric-lbl">FPS</div></div>
      <div class="metric-card"><div class="metric-val">{n}</div>
        <div class="metric-lbl">Objects</div></div>
      <div class="metric-card"><div class="metric-val">{total}</div>
        <div class="metric-lbl">Logged</div></div>
    </div>"""

def _dets_html(dets):
    if not dets:
        return '<div class="info-box">No objects above threshold.</div>'
    return "".join(
        f'<div class="det-item"><span class="det-name">🏷️ {d["name"]}</span>'
        f'<span class="det-conf">{d["confidence"]:.1f}%</span></div>'
        for d in dets
    )


# ══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "cam_running":       False,
        "camera_stream":     None,
        "camera_error":      None,
        "total_detections":  0,
        "screenshots_taken": 0,
        "last_frame":        None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

def _sidebar():
    with st.sidebar:
        st.markdown("### 🎯 DETECTION SYSTEM")
        st.divider()

        st.markdown('<div class="sec-head">⚡ Mode</div>', unsafe_allow_html=True)
        mode = st.radio("Mode", ["📹 Live Webcam", "🖼️ Image Upload"],
                        label_visibility="collapsed")

        st.markdown('<div class="sec-head">🎚 Parameters</div>', unsafe_allow_html=True)
        conf = st.slider("Confidence Threshold", 0.10, 0.95, 0.50, 0.05)
        mdl  = st.selectbox("YOLOv8 Model",
                            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0,
                            help="n=fastest · s=balanced · m=most accurate")

        st.markdown('<div class="sec-head">📈 Session</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Detected",    st.session_state.total_detections)
        c2.metric("Screenshots", st.session_state.screenshots_taken)

        st.markdown('<div class="sec-head">📂 Logs</div>', unsafe_allow_html=True)
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE) as f:
                st.download_button("📥 detection_logs.csv", f.read(),
                                   "detection_logs.csv", "text/csv")

        st.markdown("""
        <div class="info-box" style="margin-top:1rem">
          Detects <strong>80 COCO classes</strong>.<br>
          Model auto-downloads on first run (~6 MB).
        </div>""", unsafe_allow_html=True)

    return mode, conf, mdl


# ══════════════════════════════════════════════════════════════════════════
#  LIVE WEBCAM SECTION
# ══════════════════════════════════════════════════════════════════════════

def section_webcam(confidence: float, model_name: str) -> None:
    """
    Live webcam detection.

    How the UI loop works:
      - A background CameraStream thread continuously grabs frames and
        runs YOLOv8 inference, storing results under a threading.Lock.
      - The Streamlit main thread enters a  while cam_running  loop,
        reading the latest annotated frame every 80 ms and calling
        st.empty().image() to update the video placeholder in-place
        (no page reload, no flicker).
      - FPS / detection stats are updated the same way via st.empty()
        placeholders that are overwritten each iteration.
    """
    model = load_model(model_name)

    col_feed, col_stats = st.columns([2.2, 1], gap="large")

    with col_feed:
        st.markdown('<div class="sec-head">📡 Live Camera Feed</div>',
                    unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        start_btn = b1.button("▶  START",    disabled=st.session_state.cam_running)
        stop_btn  = b2.button("⏹  STOP",     disabled=not st.session_state.cam_running)
        snap_btn  = b3.button("📷  Screenshot", disabled=not st.session_state.cam_running)
        video_ph  = st.empty()

    with col_stats:
        st.markdown('<div class="sec-head">📊 Stats</div>', unsafe_allow_html=True)
        status_ph  = st.empty()
        metrics_ph = st.empty()
        dets_hdr   = st.empty()
        dets_ph    = st.empty()

    # ── START ──────────────────────────────────────────────────────────────
    if start_btn and not st.session_state.cam_running:
        cam = CameraStream(model, confidence)
        cam.start()
        st.session_state.camera_stream = cam
        st.session_state.camera_error  = None
        time.sleep(0.25)
        if cam.is_opened():
            st.session_state.cam_running = True
        else:
            st.session_state.cam_running   = False
            st.session_state.camera_stream = None
            st.session_state.camera_error  = cam.open_error()

    # ── STOP ───────────────────────────────────────────────────────────────
    if stop_btn and st.session_state.cam_running:
        if st.session_state.camera_stream:
            st.session_state.camera_stream.stop()
            st.session_state.camera_stream = None
        st.session_state.cam_running = False
        st.session_state.camera_error = None

    # ── SCREENSHOT ─────────────────────────────────────────────────────────
    if snap_btn and st.session_state.last_frame is not None:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SCREENSHOTS_DIR, f"detection_{ts}.jpg")
        cv2.imwrite(path, st.session_state.last_frame)
        st.session_state.screenshots_taken += 1
        st.toast(f"Saved → {path}", icon="📸")

    # ── LIVE LOOP ───────────────────────────────────────────────────────────
    if st.session_state.cam_running and st.session_state.camera_stream:

        # Keep confidence threshold in sync with sidebar slider
        st.session_state.camera_stream.conf_threshold = confidence

        status_ph.markdown('<span class="badge-active">● LIVE</span>',
                           unsafe_allow_html=True)
        dets_hdr.markdown('<div class="sec-head">🔍 Detections</div>',
                          unsafe_allow_html=True)

        while st.session_state.cam_running:
            frame, dets, fps = st.session_state.camera_stream.read()

            if not st.session_state.camera_stream.is_opened() and frame is None:
                st.session_state.camera_error = st.session_state.camera_stream.open_error()
                st.session_state.camera_stream.stop()
                st.session_state.camera_stream = None
                st.session_state.cam_running = False
                break

            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_ph.image(rgb, channels="RGB", use_container_width=True)

                st.session_state.last_frame       = frame
                st.session_state.total_detections += len(dets)
                _log(dets)

                metrics_ph.markdown(
                    _cards_html(fps, len(dets), st.session_state.total_detections),
                    unsafe_allow_html=True,
                )
                dets_ph.markdown(_dets_html(dets), unsafe_allow_html=True)

            else:
                video_ph.markdown(
                    '<div class="info-box">⚠️ Cannot open webcam.<br>'
                    'Check no other app is using it, then press START again.</div>',
                    unsafe_allow_html=True,
                )

            time.sleep(0.08)   # ~12 UI refreshes / sec

    else:
        # Camera is off — idle state
        status_ph.markdown('<span class="badge-inactive">○ CAMERA OFF</span>',
                           unsafe_allow_html=True)
        metrics_ph.markdown(_cards_html(0.0, 0, st.session_state.total_detections),
                            unsafe_allow_html=True)
        dets_hdr.markdown('<div class="sec-head">🔍 Detections</div>',
                          unsafe_allow_html=True)
        dets_ph.markdown(
            '<div class="info-box">Press <strong>▶ START</strong> to begin.</div>',
            unsafe_allow_html=True,
        )
        video_ph.markdown(
            '<div class="info-box" style="padding:3rem;text-align:center;">'
            '🎥 Camera feed will appear here</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════
#  IMAGE UPLOAD SECTION
# ══════════════════════════════════════════════════════════════════════════

def section_image(confidence: float, model_name: str) -> None:
    """Static image detection — upload → infer → display side-by-side."""
    model = load_model(model_name)

    col_orig, col_res = st.columns(2, gap="large")

    with col_orig:
        st.markdown('<div class="sec-head">📤 Upload Image</div>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop an image here",
                                    type=["jpg","jpeg","png","bmp","webp"])
        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Original", use_container_width=True)

    with col_res:
        st.markdown('<div class="sec-head">🎯 Result</div>', unsafe_allow_html=True)

        if not uploaded:
            st.markdown(
                '<div class="info-box">Upload an image on the left.</div>',
                unsafe_allow_html=True)
            return

        with st.spinner("Running YOLOv8 …"):
            img_bgr  = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            t0       = time.perf_counter()
            results  = model(img_bgr, verbose=False)
            ms       = (time.perf_counter() - t0) * 1000
            ann_bgr, detections = _draw_detections(img_bgr, results, confidence)
            ann_rgb  = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

        st.image(ann_rgb, caption="Detected Objects", use_container_width=True)

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card"><div class="metric-val">{len(detections)}</div>
            <div class="metric-lbl">Objects</div></div>
          <div class="metric-card"><div class="metric-val">{ms:.0f}</div>
            <div class="metric-lbl">ms</div></div>
          <div class="metric-card"><div class="metric-val">{confidence:.0%}</div>
            <div class="metric-lbl">Min Conf</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-head">🔍 Detected Objects</div>',
                    unsafe_allow_html=True)
        st.markdown(_dets_html(detections), unsafe_allow_html=True)

        if detections:
            buf = io.BytesIO()
            Image.fromarray(ann_rgb).save(buf, format="PNG")
            st.download_button("📥 Download Annotated Image", buf.getvalue(),
                               f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                               "image/png")
        _log(detections)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    _init_state()
    _init_log()

    st.markdown(
        '<h1 class="main-title">🎯 REAL-TIME OBJECT DETECTION SYSTEM</h1>',
        unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">YOLOv8 · OpenCV · Streamlit · COCO 80 Classes</p>',
        unsafe_allow_html=True)

    mode, confidence, model_name = _sidebar()

    if mode == "📹 Live Webcam":
        section_webcam_webrtc(
            confidence,
            model_name,
            load_model=load_model,
            _cards_html=_cards_html,
            _dets_html=_dets_html,
            _draw_detections=_draw_detections,
            _draw_hud=_draw_hud,
        )
    else:
        section_image(confidence, model_name)


if __name__ == "__main__":
    main()
