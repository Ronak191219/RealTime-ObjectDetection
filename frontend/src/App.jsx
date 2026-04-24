// filepath: frontend/src/App.jsx
import { useState, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import LiveWebcam from './components/LiveWebcam'
import ImageUpload from './components/ImageUpload'

function App() {
  const [mode, setMode] = useState('live') // 'live' or 'image'
  const [confidence, setConfidence] = useState(0.5)
  const [model, setModel] = useState('yolov8n.pt')
  const [totalDetections, setTotalDetections] = useState(0)
  const [screenshotsTaken, setScreenshotsTaken] = useState(0)

  return (
    <div className="app">
      <Sidebar 
        mode={mode}
        setMode={setMode}
        confidence={confidence}
        setConfidence={setConfidence}
        model={model}
        setModel={setModel}
        totalDetections={totalDetections}
        screenshotsTaken={screenshotsTaken}
      />
      
      <main className="main-content">
        <header className="app-header">
          <h1 className="main-title">🎯 Real-Time Object Detection System</h1>
          <p className="subtitle">YOLOv8 · OpenCV · React · FastAPI</p>
        </header>

        <div className="content-area">
          {mode === 'live' ? (
            <LiveWebcam 
              confidence={confidence}
              model={model}
              onDetection={(count) => setTotalDetections(prev => prev + count)}
              onScreenshot={() => setScreenshotsTaken(prev => prev + 1)}
            />
          ) : (
            <ImageUpload 
              confidence={confidence}
              model={model}
              onDetection={(count) => setTotalDetections(prev => prev + count)}
            />
          )}
        </div>
      </main>
    </div>
  )
}

export default App