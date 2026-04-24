# filepath: backend/app/logging.py
"""
CSV Logging Module
Based on the original Streamlit app.py implementation
"""

import os
import csv
from datetime import datetime
from typing import List, Dict


# Constants
LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, "detection_logs.csv")

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)


def init_log():
    """Create CSV with headers if it doesn't exist yet."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "object_name", "confidence_%",
                 "x1", "y1", "x2", "y2"]
            )


def log_detections(detections: List[Dict]):
    """Append detections to CSV log with timestamp."""
    if not detections:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        for d in detections:
            w.writerow([ts, d["name"], d["confidence"],
                        d["x1"], d["y1"], d["x2"], d["y2"]])


def get_log_file_path() -> str:
    """Return the log file path."""
    return LOG_FILE


def get_log_file_contents() -> str:
    """Return the contents of the log file."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return f.read()
    return ""