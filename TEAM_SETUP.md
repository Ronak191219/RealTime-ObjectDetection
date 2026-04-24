# Team Setup Guide

This guide explains what a teammate should do after pulling the latest code.

## Project Structure

- `backend/` = FastAPI + YOLOv8
- `frontend/` = React + Vite
- `venv/` = Python virtual environment used by the backend scripts

The app now runs as two separate services in two different terminals.

## Prerequisites

- Python 3.10+ installed
- Node.js and npm installed
- Webcam available if you want to test live detection

## One-Time Setup After Pull

Open a terminal in the project root:

```powershell
cd RealTime-ObjectDetection
```

### 1. Create the Python virtual environment

If `venv/` does not already exist:

```powershell
python -m venv venv
```

### 2. Install backend dependencies

```powershell
cd backend
npm run setup
```

What this does:

- Uses `..\venv\Scripts\python.exe`
- Installs everything from `backend/requirements.txt`

Important:
Do not use global `pip install ...` for backend dependencies. Use `npm run setup` so packages are installed into the project `venv`.

### 3. Install frontend dependencies

Open another terminal or continue after backend setup:

```powershell
cd ..\frontend
npm install
```

## How To Start The Project

You need two terminals.

### Terminal 1: Start Backend

```powershell
cd RealTime-ObjectDetection\backend
npm run dev
```

Backend runs on:

- `http://localhost:8000`
- FastAPI docs: `http://localhost:8000/docs`

### Terminal 2: Start Frontend

```powershell
cd RealTime-ObjectDetection\frontend
npm run dev
```

Frontend usually runs on:

- `http://localhost:5173`

Then open the frontend URL in the browser.

## Normal Daily Workflow

After the first setup, teammates usually only need:

```powershell
cd RealTime-ObjectDetection\backend
npm run dev
```

and in another terminal:

```powershell
cd RealTime-ObjectDetection\frontend
npm run dev
```

## If Someone Pulls New Dependency Changes

Run these again:

```powershell
cd RealTime-ObjectDetection\backend
npm run setup
```

```powershell
cd RealTime-ObjectDetection\frontend
npm install
```

Use this when:

- `backend/requirements.txt` changed
- `frontend/package.json` changed

## Common Issues

### 1. `No module named uvicorn`

Cause:
Backend packages were installed with global `pip` instead of the project virtual environment.

Fix:

```powershell
cd RealTime-ObjectDetection\backend
npm run setup
```

### 2. `react`, `react-dom`, or `react/jsx-dev-runtime` could not be resolved

Cause:
Frontend dependencies were not installed yet, or `node_modules` is stale.

Fix:

```powershell
cd RealTime-ObjectDetection\frontend
npm install
```

If needed:

```powershell
Remove-Item -Recurse -Force node_modules
Remove-Item -Force package-lock.json
npm install
```

### 3. Frontend starts on `5174` instead of `5173`

Cause:
Port `5173` is already in use.

Fix:

- Close the app using `5173`, or
- Use the Vite URL shown in the terminal

Note:
The backend currently allows `http://localhost:5173` and `http://127.0.0.1:5173` by default, so keeping the frontend on `5173` is the safest option.

### 4. Webcam not opening

Cause:
Another app is already using the camera.

Fix:

- Close Zoom, Teams, Discord, browser camera tabs, or other webcam apps
- Restart the backend if needed

## Quick Start Summary

First time after pull:

```powershell
cd RealTime-ObjectDetection\backend
npm run setup
cd ..\frontend
npm install
```

Run project:

```powershell
cd RealTime-ObjectDetection\backend
npm run dev
```

```powershell
cd RealTime-ObjectDetection\frontend
npm run dev
```

## Current App URLs

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- Backend docs: `http://localhost:8000/docs`
