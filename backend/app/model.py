# filepath: backend/app/model.py
"""
YOLOv8 Model Loading Module
Based on the original Streamlit app.py implementation
"""

from ultralytics import YOLO


# Model cache - loads once and persists
_model_cache = {}


def load_model(name: str = "yolov8n.pt") -> YOLO:
    """
    Load YOLOv8 weights. Auto-downloads from Ultralytics on first run (~6 MB).
    Cached to avoid reloading on every request.
    
    Args:
        name: Model name (yolov8n.pt, yolov8s.pt, yolov8m.pt)
    
    Returns:
        YOLO model instance
    """
    if name not in _model_cache:
        _model_cache[name] = YOLO(name)
    return _model_cache[name]


def get_available_models() -> list:
    """Return list of available YOLOv8 models."""
    return [
        {"id": "yolov8n.pt", "name": "YOLOv8 Nano", "description": "Fastest, smallest"},
        {"id": "yolov8s.pt", "name": "YOLOv8 Small", "description": "Balanced"},
        {"id": "yolov8m.pt", "name": "YOLOv8 Medium", "description": "Most accurate"},
    ]