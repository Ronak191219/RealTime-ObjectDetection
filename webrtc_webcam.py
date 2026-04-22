import threading

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer


RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}


def section_webcam_webrtc(
    confidence,
    model_name,
    *,
    load_model,
    _cards_html,
    _dets_html,
    _draw_detections,
    _draw_hud,
):
    """
    Live webcam mode backed by browser WebRTC.

    The browser requests camera permission itself, matching the behavior
    users expect from apps like Google Meet.
    """
    model = load_model(model_name)

    class ObjectDetectorProcessor:
        def __init__(self):
            self.lock = threading.Lock()
            self.fps = 0.0
            self.detections = []
            self.frame_count = 0
            self.fps_timer = None

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img, verbose=False)
            annotated, detections = _draw_detections(img, results, confidence)

            if self.fps_timer is None:
                self.fps_timer = cv2.getTickCount() / cv2.getTickFrequency()

            self.frame_count += 1
            now = cv2.getTickCount() / cv2.getTickFrequency()
            elapsed = now - self.fps_timer
            current_fps = self.fps
            if elapsed >= 1.0:
                current_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_timer = now

            annotated = _draw_hud(annotated, current_fps, len(detections))

            with self.lock:
                self.fps = current_fps
                self.detections = detections

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    col_feed, col_stats = st.columns([2.2, 1], gap="large")

    with col_feed:
        st.markdown('<div class="sec-head">Live Camera Feed</div>',
                    unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        start_btn = b1.button("START", disabled=st.session_state.cam_running)
        stop_btn = b2.button("STOP", disabled=not st.session_state.cam_running)

    if start_btn:
        st.session_state.cam_running = True
        st.session_state.camera_error = None

    if stop_btn:
        st.session_state.cam_running = False
        st.session_state.camera_error = None

    with col_feed:
        st.markdown(
            '<div class="info-box">When START is pressed, Chrome should ask for camera permission. Click <strong>Allow</strong> to open the live stream.</div>',
            unsafe_allow_html=True,
        )

        webrtc_ctx = webrtc_streamer(
            key="live-object-detection",
            mode=WebRtcMode.SENDRECV,
            desired_playing_state=st.session_state.cam_running,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=ObjectDetectorProcessor,
            async_processing=True,
            sendback_audio=False,
        )

    with col_stats:
        st.markdown('<div class="sec-head">Stats</div>', unsafe_allow_html=True)
        playing = bool(webrtc_ctx and webrtc_ctx.state.playing)
        if playing:
            st.markdown('<span class="badge-active">LIVE</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-inactive">CAMERA OFF</span>',
                        unsafe_allow_html=True)

        fps = 0.0
        detections = []
        if webrtc_ctx and webrtc_ctx.video_processor:
            with webrtc_ctx.video_processor.lock:
                fps = webrtc_ctx.video_processor.fps
                detections = list(webrtc_ctx.video_processor.detections)

        st.markdown(
            _cards_html(fps, len(detections), st.session_state.total_detections),
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sec-head">Detections</div>',
                    unsafe_allow_html=True)
        st.markdown(_dets_html(detections), unsafe_allow_html=True)

        if playing and detections:
            st.session_state.total_detections += len(detections)
