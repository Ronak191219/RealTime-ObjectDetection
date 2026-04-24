# filepath: backend/app/routes/detect.py
"""
Detection Routes - Live webcam and image detection endpoints
"""

import time
import io
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from PIL import Image

from app.model import load_model
from app.camera import get_camera, release_camera
from app.detection import draw_detections, draw_hud, encode_frame_to_jpeg
from app.logging import log_detections


router = APIRouter(prefix="/api/detect", tags=["Detection"])

# Global state
_current_model = "yolov8n.pt"
_current_conf_threshold = 0.5
_frame_count = 0
_fps_timer = time.time()
_current_fps = 0.0


@router.get("/models")
async def get_models():
    """Get available YOLOv8 models."""
    from app.model import get_available_models
    return {"models": get_available_models()}


@router.post("/start")
async def start_camera(
    model: str = Form("yolov8n.pt"),
    confidence: float = Form(0.5)
):
    """Start the camera for live detection."""
    global _current_model, _current_conf_threshold
    
    _current_model = model
    _current_conf_threshold = confidence
    
    camera = get_camera()
    
    if not camera.open():
        raise HTTPException(status_code=500, detail="Cannot open webcam. Make sure no other app is using it.")
    
    return {"status": "started", "model": model, "confidence": confidence}


@router.post("/stop")
async def stop_camera():
    """Stop the camera."""
    release_camera()
    return {"status": "stopped"}


@router.get("/live")
async def live_detection():
    """
    MJPEG streaming endpoint for live webcam detection.
    Returns multipart/x-mixed-replace with JPEG frames.
    """
    camera = get_camera()
    
    if not camera.is_opened():
        raise HTTPException(status_code=400, detail="Camera not started. Call /api/detect/start first.")
    
    model = load_model(_current_model)
    camera.conf_threshold = _current_conf_threshold
    
    def generate_frames():
        global _frame_count, _fps_timer, _current_fps
        
        while camera.is_opened():
            frame = camera.read()
            
            if frame is None:
                break
            
            # Run inference
            results = model(frame, verbose=False)
            annotated, dets = draw_detections(frame, results, _current_conf_threshold)
            
            # Calculate FPS
            _frame_count += 1
            elapsed = time.time() - _fps_timer
            if elapsed >= 1.0:
                _current_fps = _frame_count / elapsed
                _frame_count = 0
                _fps_timer = time.time()
            
            # Draw HUD
            annotated = draw_hud(annotated, _current_fps, len(dets))
            
            # Log detections
            log_detections(dets)
            
            # Encode to JPEG
            frame_bytes = encode_frame_to_jpeg(annotated)
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.post("/image")
async def detect_image(
    file: UploadFile = File(...),
    model: str = Form("yolov8n.pt"),
    confidence: float = Form(0.5)
):
    """
    Detect objects in an uploaded image.
    
    Returns:
        JSON with detections array and annotated image as base64
    """
    global _current_model, _current_conf_threshold
    
    _current_model = model
    _current_conf_threshold = confidence
    
    # Read and convert image
    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Run inference
    model_instance = load_model(model)
    t0 = time.perf_counter()
    results = model_instance(img_bgr, verbose=False)
    ms = (time.perf_counter() - t0) * 1000
    
    # Draw detections
    ann_bgr, detections = draw_detections(img_bgr, results, confidence)
    
    # Convert back to RGB for response
    ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
    
    # Encode annotated image to base64
    import base64
    buffered = io.BytesIO()
    Image.fromarray(ann_rgb).save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Log detections
    log_detections(detections)
    
    return {
        "detections": detections,
        "count": len(detections),
        "inference_time_ms": round(ms, 2),
        "image": f"data:image/png;base64,{img_base64}"
    }


@router.get("/stats")
async def get_stats():
    """Get current detection stats."""
    return {
        "fps": round(_current_fps, 1),
        "model": _current_model,
        "confidence_threshold": _current_conf_threshold
    }