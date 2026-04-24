// filepath: frontend/src/components/ImageUpload.jsx
import { useState, useRef } from 'react'

function ImageUpload({ confidence, model, onDetection }) {
  const [originalImage, setOriginalImage] = useState(null)
  const [annotatedImage, setAnnotatedImage] = useState(null)
  const [detections, setDetections] = useState([])
  const [inferenceTime, setInferenceTime] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleFileChange = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    setError(null)
    
    // Show original image
    const reader = new FileReader()
    reader.onload = (event) => {
      setOriginalImage(event.target.result)
    }
    reader.readAsDataURL(file)

    // Run detection
    await runDetection(file)
  }

  const runDetection = async (file) => {
    setIsLoading(true)
    setAnnotatedImage(null)
    setDetections([])
    
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('model', model)
      formData.append('confidence', confidence)

      const response = await fetch('/api/detect/image', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Detection failed')
      }

      const data = await response.json()
      
      setAnnotatedImage(data.image)
      setDetections(data.detections)
      setInferenceTime(data.inference_time_ms)
      onDetection(data.count)
      
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      handleFileChange({ target: { files: [file] } })
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
  }

  return (
    <div style={{ display: 'flex', gap: '1.5rem', width: '100%' }}>
      {/* Left Column - Upload */}
      <div className="video-column">
        <div className="content-section-header">📤 UPLOAD IMAGE</div>
        
        <div 
          className="upload-area"
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/jpg,image/png,image/bmp,image/webp"
            onChange={handleFileChange}
          />
          <div className="upload-icon">📁</div>
          <div className="upload-text">
            Drag & drop or click to upload<br />
            <small>Supports JPG, JPEG, PNG, BMP, WebP</small>
          </div>
        </div>

        {originalImage && (
          <div className="image-panel" style={{ marginTop: '1rem' }}>
            <div className="content-section-header">🖼️ ORIGINAL</div>
            <img src={originalImage} alt="Original" />
          </div>
        )}
      </div>

      {/* Right Column - Results */}
      <div className="image-column">
        <div className="content-section-header">🎯 DETECTED OBJECTS</div>

        {!originalImage && (
          <div className="info-box">
            📁 Upload an image on the left to run YOLOv8 detection.
          </div>
        )}

        {isLoading && (
          <div style={{ textAlign: 'center', padding: '2rem' }}>
            <div className="spinner"></div>
            <p style={{ marginTop: '1rem', color: '#3a5a70' }}>
              Running YOLOv8 inference …
            </p>
          </div>
        )}

        {error && (
          <div className="info-box" style={{ color: '#ff6b6b', marginTop: '1rem' }}>
            ⚠️ {error}
          </div>
        )}

        {annotatedImage && !isLoading && (
          <>
            <div className="content-section-header">✅ RESULT</div>
            <img src={annotatedImage} alt="Detected objects" />
            
            {/* Stats */}
            <div className="metric-row" style={{ marginTop: '1rem' }}>
              <div className="metric-card">
                <div className="metric-val">{detections.length}</div>
                <div className="metric-lbl">Objects</div>
              </div>
              <div className="metric-card">
                <div className="metric-val">{Math.round(inferenceTime)}</div>
                <div className="metric-lbl">MS</div>
              </div>
              <div className="metric-card">
                <div className="metric-val">{Math.round(confidence * 100)}%</div>
                <div className="metric-lbl">Min Conf</div>
              </div>
            </div>

            {/* Detection List */}
            <div className="detection-list">
              {detections.map((det, idx) => (
                <div key={idx} className="det-item">
                  <span className="det-name">🏷️ {det.name}</span>
                  <span className="det-conf">{det.confidence.toFixed(1)}%</span>
                </div>
              ))}
            </div>

            {/* Download Button */}
            {detections.length > 0 && (
              <a 
                href={annotatedImage}
                download={`detection_${Date.now()}.png`}
                className="btn"
                style={{ 
                  display: 'block', 
                  textAlign: 'center', 
                  marginTop: '1rem',
                  textDecoration: 'none'
                }}
              >
                📥 Download Annotated Image
              </a>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default ImageUpload