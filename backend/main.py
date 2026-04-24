# filepath: backend/main.py
"""
FastAPI Backend for Real-Time Object Detection
YOLOv8 + OpenCV + FastAPI
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes import detect, logs, screenshots
from app.logging import init_log


# Create necessary directories
os.makedirs("screenshots", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Initialize logging
init_log()

# Create FastAPI app
app = FastAPI(
    title="Real-Time Object Detection API",
    description="YOLOv8 Object Detection with FastAPI backend",
    version="1.0.0"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(detect.router)
app.include_router(logs.router)
app.include_router(screenshots.router)

# Serve static files (screenshots)
app.mount("/api/screenshots/files", StaticFiles(directory="screenshots"), name="screenshots")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Real-Time Object Detection API",
        "docs": "/docs",
        "endpoints": {
            "detection": "/api/detect",
            "logs": "/api/logs",
            "screenshots": "/api/screenshots"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)