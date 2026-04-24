// filepath: frontend/src/components/Sidebar.jsx
import { useState, useEffect } from 'react'

function Sidebar({ 
  mode, 
  setMode, 
  confidence, 
  setConfidence, 
  model, 
  setModel,
  totalDetections,
  screenshotsTaken
}) {
  const [logsAvailable, setLogsAvailable] = useState(false)

  useEffect(() => {
    // Check if logs exist
    fetch('/api/logs/json')
      .then(res => res.json())
      .then(data => setLogsAvailable(data.data && data.data.length > 0))
      .catch(() => setLogsAvailable(false))
  }, [])

  const downloadLogs = async () => {
    try {
      const response = await fetch('/api/logs')
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'detection_logs.csv'
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error('Error downloading logs:', error)
    }
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <span>🎯 DETECTION SYSTEM</span>
      </div>

      <hr />

      {/* Mode Selection */}
      <div className="sidebar-section">
        <div className="section-header">⚡ MODE</div>
        <div className="radio-group">
          <label className={`radio-option ${mode === 'live' ? 'active' : ''}`}>
            <input 
              type="radio" 
              name="mode" 
              value="live"
              checked={mode === 'live'}
              onChange={() => setMode('live')}
            />
            📹 Live Webcam
          </label>
          <label className={`radio-option ${mode === 'image' ? 'active' : ''}`}>
            <input 
              type="radio" 
              name="mode" 
              value="image"
              checked={mode === 'image'}
              onChange={() => setMode('image')}
            />
            🖼️ Image Upload
          </label>
        </div>
      </div>

      {/* Parameters */}
      <div className="sidebar-section">
        <div className="section-header">🎚 PARAMETERS</div>
        
        <div className="slider-container">
          <label>Confidence Threshold</label>
          <input 
            type="range" 
            min="0.10" 
            max="0.95" 
            step="0.05"
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
          />
          <div className="slider-value">{Math.round(confidence * 100)}%</div>
        </div>

        <div className="select-container" style={{ marginTop: '1rem' }}>
          <label>YOLOv8 Model</label>
          <select 
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            <option value="yolov8n.pt">YOLOv8 Nano (Fastest)</option>
            <option value="yolov8s.pt">YOLOv8 Small (Balanced)</option>
            <option value="yolov8m.pt">YOLOv8 Medium (Accurate)</option>
          </select>
        </div>
      </div>

      <hr />

      {/* Session Stats */}
      <div className="sidebar-section">
        <div className="section-header">📈 SESSION</div>
        <div className="metrics-row">
          <div className="metric-box">
            <div className="value">{totalDetections}</div>
            <div className="label">Detected</div>
          </div>
          <div className="metric-box">
            <div className="value">{screenshotsTaken}</div>
            <div className="label">Screenshots</div>
          </div>
        </div>
      </div>

      <hr />

      {/* Logs Download */}
      <div className="sidebar-section">
        <div className="section-header">📂 LOGS</div>
        <button 
          className="download-btn"
          onClick={downloadLogs}
          disabled={!logsAvailable}
        >
          📥 detection_logs.csv
        </button>
        {!logsAvailable && (
          <p style={{ fontSize: '0.75rem', color: '#3a5a70', marginTop: '0.5rem' }}>
            No logs yet. Run a detection first.
          </p>
        )}
      </div>

      <hr />

      {/* Info */}
      <div className="info-box">
        Detects <strong style={{ color: '#00d4ff' }}>80 COCO classes</strong> —
        people, vehicles, animals, everyday objects &amp; more.<br /><br />
        Model weights auto-download from Ultralytics on first run (~6 MB).
      </div>
    </aside>
  )
}

export default Sidebar