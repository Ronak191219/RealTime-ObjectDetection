# filepath: backend/app/camera.py
"""
Camera Stream Module
Based on the original Streamlit app.py implementation
"""

import cv2


class CameraStream:
    """
    Thin wrapper for webcam capture.
    Actual capture happens in the main thread to avoid
    Windows threading issues with cv2.VideoCapture.
    """
    
    def __init__(self):
        self.running = False
        self.cap = None
        self.conf_threshold = 0.5

    def open(self) -> bool:
        """
        Open webcam in the main thread — required on Windows.
        Tries camera indices 0-2.
        
        Returns:
            True if camera opened successfully
        """
        for idx in range(3):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap = cap
                    self.running = True
                    return True
            cap.release()
        return False

    def read(self):
        """Read a frame from the camera."""
        if self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            return frame if ok else None
        return None

    def stop(self):
        """Stop and release the camera."""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def is_opened(self) -> bool:
        """Check if camera is currently open."""
        return self.cap is not None and self.cap.isOpened()


# Global camera instance
_camera = None


def get_camera() -> CameraStream:
    """Get or create the global camera instance."""
    global _camera
    if _camera is None:
        _camera = CameraStream()
    return _camera


def release_camera():
    """Release the global camera."""
    global _camera
    if _camera:
        _camera.stop()
        _camera = None