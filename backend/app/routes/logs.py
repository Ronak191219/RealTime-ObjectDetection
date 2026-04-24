# filepath: backend/app/routes/logs.py
"""
Logs Routes - CSV log download endpoint
"""

import os
from fastapi import APIRouter
from fastapi.responses import Response

from app.logging import get_log_file_path, get_log_file_contents


router = APIRouter(prefix="/api", tags=["Logs"])


@router.get("/logs")
async def get_logs():
    """Get detection logs as CSV file."""
    contents = get_log_file_contents()
    
    if not contents:
        return {"message": "No logs yet", "data": []}
    
    return Response(
        content=contents,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=detection_logs.csv"}
    )


@router.get("/logs/json")
async def get_logs_json():
    """Get detection logs as JSON."""
    log_file = get_log_file_path()
    
    if not os.path.exists(log_file):
        return {"data": []}
    
    import csv
    data = []
    with open(log_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    return {"data": data}