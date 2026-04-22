import os
import time
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image


def section_webcam_browser(
    confidence,
    model_name,
    *,
    load_model,
    _cards_html,
    _dets_html,
    _draw_detections,
    _log,
    SCREENSHOTS_DIR,
):
    """
    Browser camera mode.

    START reveals a browser camera widget so the browser can request
    camera permission directly from the user.
    """
    model = load_model(model_name)

    col_feed, col_stats = st.columns([2.2, 1], gap="large")

    with col_feed:
        st.markdown('<div class="sec-head">Browser Camera</div>',
                    unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        start_btn = b1.button("START", disabled=st.session_state.cam_running)
        stop_btn = b2.button("STOP", disabled=not st.session_state.cam_running)
        snap_btn = b3.button("Screenshot", disabled=not st.session_state.cam_running)
        video_ph = st.empty()

    with col_stats:
        st.markdown('<div class="sec-head">Stats</div>', unsafe_allow_html=True)
        status_ph = st.empty()
        metrics_ph = st.empty()
        dets_hdr = st.empty()
        dets_ph = st.empty()

    if start_btn and not st.session_state.cam_running:
        st.session_state.cam_running = True
        st.session_state.camera_error = None

    if stop_btn and st.session_state.cam_running:
        st.session_state.cam_running = False
        st.session_state.camera_error = None

    if snap_btn and st.session_state.last_frame is not None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SCREENSHOTS_DIR, f"detection_{ts}.jpg")
        cv2.imwrite(path, st.session_state.last_frame)
        st.session_state.screenshots_taken += 1
        st.toast(f"Saved -> {path}")

    if st.session_state.cam_running:
        status_ph.markdown('<span class="badge-active">LIVE</span>',
                           unsafe_allow_html=True)
        dets_hdr.markdown('<div class="sec-head">Detections</div>',
                          unsafe_allow_html=True)

        camera_frame = st.camera_input("Browser Camera", key="browser_camera_input")

        if camera_frame is None:
            video_ph.markdown(
                '<div class="info-box">Click Allow in your browser when it asks for camera permission, then capture a frame.</div>',
                unsafe_allow_html=True,
            )
            metrics_ph.markdown(
                _cards_html(0.0, 0, st.session_state.total_detections),
                unsafe_allow_html=True,
            )
            dets_ph.markdown(
                '<div class="info-box">Waiting for permission and captured frame.</div>',
                unsafe_allow_html=True,
            )
            return

        pil_img = Image.open(camera_frame).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        t0 = time.perf_counter()
        results = model(img_bgr, verbose=False)
        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        ann_bgr, detections = _draw_detections(img_bgr, results, confidence)
        ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

        video_ph.image(ann_rgb, channels="RGB", use_container_width=True)
        st.session_state.last_frame = ann_bgr
        st.session_state.total_detections += len(detections)
        _log(detections)

        metrics_ph.markdown(
            _cards_html(fps, len(detections), st.session_state.total_detections),
            unsafe_allow_html=True,
        )
        dets_ph.markdown(_dets_html(detections), unsafe_allow_html=True)
    else:
        status_ph.markdown('<span class="badge-inactive">CAMERA OFF</span>',
                           unsafe_allow_html=True)
        metrics_ph.markdown(_cards_html(0.0, 0, st.session_state.total_detections),
                            unsafe_allow_html=True)
        dets_hdr.markdown('<div class="sec-head">Detections</div>',
                          unsafe_allow_html=True)
        dets_ph.markdown(
            '<div class="info-box">Press <strong>START</strong> to open the browser camera.</div>',
            unsafe_allow_html=True,
        )
        video_ph.markdown(
            '<div class="info-box" style="padding:3rem;text-align:center;">Camera preview will appear here after permission is granted.</div>',
            unsafe_allow_html=True,
        )
