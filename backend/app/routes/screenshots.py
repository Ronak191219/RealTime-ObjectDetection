# filepath: backend/app/routes/screenshots.py
"""
Screenshots Routes - Save and retrieve screenshots
"""

import os
from datetime import datetime
from typing import List

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse

from app.detection import encode_frame_to_jpeg


router = APIRouter(prefix="/api/screenshots", tags=["Screenshots"])

# Constants
SCREENSHOTS_DIR = "screenshots"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


@router.post("")
async def save_screenshot(
    file: UploadFile = File(...)
):
    """Save a screenshot."""
    contents = await file.read()
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detection_{ts}.jpg"
    filepath = os.path.join(SCREENSHOTS_DIR, filename)
    
    with open(filepath, "wb") as f:
        f.write(contents)
    
    return {"status": "saved", "filename": filename, "path": filepath}


@router.get("")
async def list_screenshots():
    """List all saved screenshots."""
    files = []
    for f in os.listdir(SCREENSHOTS_DIR):
        if f.endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(SCREENSHOTS_DIR, f)
            files.append({
                "name": f,
                "path": f"/api/screenshots/files/{f}",
                "size": os.path.getsize(filepath),
                "created": datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
            })
    return {"screenshots": files}


@router.get("/files/{filename}")
async def get_screenshot(filename: str):
    """Get a specific screenshot file."""
    filepath = os.path.join(SCREENSHOTS_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Screenshot not found")
    
    return FileResponse(filepath, media_type="image/jpeg")


@router.delete("/{filename}")
async def delete_screenshot(filename: str):
    """Delete a screenshot."""
    filepath = os.path.join(SCREENSHOTS_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Screenshot not found")
    
    os.remove(filepath)
    return {"status": "deleted", "filename": filename}