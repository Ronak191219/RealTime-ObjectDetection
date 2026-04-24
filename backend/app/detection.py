# filepath: backend/app/detection.py
"""
Detection Module - Drawing helpers and detection processing
Based on the original Streamlit app.py implementation
"""

import numpy as np
import cv2


# Fixed consistent color per COCO class (same color = same object always)
np.random.seed(42)
CLASS_COLORS = np.random.randint(60, 220, size=(80, 3), dtype=np.uint8).tolist()


def draw_detections(frame: np.ndarray, results, conf_thresh: float):
    """
    Parse YOLO results and draw bounding boxes + label pills on the frame.

    Args:
        frame: Input frame (BGR)
        results: YOLO results object
        conf_thresh: Confidence threshold

    Returns:
        annotated_frame (np.ndarray), detections (list of dict)
    """
    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            # Confidence filter
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue

            # Extract class info
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]

            # Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Color for this class
            color = tuple(CLASS_COLORS[cls_id % 80])

            # Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label Pill
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

            # Store result
            detections.append({
                "name": cls_name,
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


def encode_frame_to_jpeg(frame: np.ndarray) -> bytes:
    """Encode a frame to JPEG bytes."""
    ret, buffer = cv2.imencode('.jpg', frame)
    if ret:
        return buffer.tobytes()
    return b''