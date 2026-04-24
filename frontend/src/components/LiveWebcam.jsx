// filepath: frontend/src/components/LiveWebcam.jsx
import { useState, useEffect, useRef } from 'react'

function LiveWebcam({ confidence, model, onDetection, onScreenshot }) {
  const [isRunning, setIsRunning] = useState(false)
  const [detections, setDetections] = useState([])
  const [fps, setFps] = useState(0)
  const [error, setError] = useState(null)
  const imgRef = useRef(null)
  const streamRef = useRef(null)

  const startCamera = async () => {
    try {
      setError(null)
      const response = await fetch('/api/detect/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `model=${model}&confidence=${confidence}`
      })
      
      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to start camera')
      }
      
      setIsRunning(true)
    } catch (err) {
      setError(err.message)
    }
  }

  const stopCamera = async () => {
    try {
      await fetch('/api/detect/stop', { method: 'POST' })
      setIsRunning(false)
      setDetections([])
      setFps(0)
    } catch (err) {
      console.error('Error stopping camera:', err)
    }
  }

  const takeScreenshot = async () => {
    if (!imgRef.current) return
    
    try {
      const canvas = document.createElement('canvas')
      const img = imgRef.current
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0)
      
      canvas.toBlob(async (blob) => {
        const formData = new FormData()
        formData.append('file', blob, 'screenshot.jpg')
        
        await fetch('/api/screenshots', {
          method: 'POST',
          body: formData
        })
        
        onScreenshot()
      }, 'image/jpeg')
    } catch (err) {
      console.error('Error taking screenshot:', err)
    }
  }

  // Set up MJPEG stream when running
  useEffect(() => {
    if (isRunning && imgRef.current) {
      imgRef.current.src = '/api/detect/live'
      
      // Poll for stats
      const statsInterval = setInterval(async () => {
        try {
          const res = await fetch('/api/detect/stats')
          const data = await res.json()
          setFps(data.fps)
        } catch (e) {}
      }, 500)
      
      return () => {
        clearInterval(statsInterval)
        if (imgRef.current) {
          imgRef.current.src = ''
        }
      }
    }
  }, [isRunning])

  // Update camera settings when changed
  useEffect(() => {
    if (isRunning) {
      // Restart with new settings
      stopCamera().then(() => startCamera())
    }
  }, [model, confidence])

  return (
    <div style={{ display: 'flex', gap: '1.5rem', width: '100%' }}>
      {/* Video Column */}
      <div className="video-column">
        <div className="content-section-header">📡 LIVE CAMERA FEED</div>
        
        <div className="btn-group">
          <button 
            className="btn start"
            onClick={startCamera}
            disabled={isRunning}
          >
            ▶ START
          </button>
          <button 
            className="btn stop"
            onClick={stopCamera}
            disabled={!isRunning}
          >
            ⏹ STOP
          </button>
          <button 
            className="btn"
            onClick={takeScreenshot}
            disabled={!isRunning}
          >
            📷 Screenshot
          </button>
        </div>

        <div className="video-container">
          {isRunning ? (
            <img ref={imgRef} alt="Live camera feed" />
          ) : (
            <div className="video-placeholder">
              🎥<br /><br />
              Camera feed will appear here
            </div>
          )}
        </div>
        
        {error && (
          <div className="info-box" style={{ marginTop: '1rem', color: '#ff6b6b' }}>
            ⚠️ {error}
          </div>
        )}
      </div>

      {/* Stats Column */}
      <div className="image-column">
        <div className="content-section-header">📊 LIVE STATS</div>
        
        {/* Status */}
        <div style={{ marginBottom: '1rem' }}>
          {isRunning ? (
            <span className="status-badge live">● &nbsp;LIVE</span>
          ) : (
            <span className="status-badge off">○ &nbsp;CAMERA OFF</span>
          )}
        </div>

        {/* Metrics */}
        <div className="metric-row">
          <div className="metric-card">
            <div className="metric-val">{fps.toFixed(1)}</div>
            <div className="metric-lbl">FPS</div>
          </div>
          <div className="metric-card">
            <div className="metric-val">{detections.length}</div>
            <div className="metric-lbl">Objects</div>
          </div>
        </div>

        {/* Detection List */}
        <div className="content-section-header" style={{ marginTop: '1rem' }}>
          🔍 DETECTED OBJECTS
        </div>
        
        {detections.length > 0 ? (
          <div className="detection-list">
            {detections.map((det, idx) => (
              <div key={idx} className="det-item">
                <span className="det-name">🏷️ {det.name}</span>
                <span className="det-conf">{det.confidence.toFixed(1)}%</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="info-box">
            {isRunning 
              ? 'No objects detected above threshold.' 
              : 'Press ▶ START to activate the camera.'}
          </div>
        )}
      </div>
    </div>
  )
}

export default LiveWebcam