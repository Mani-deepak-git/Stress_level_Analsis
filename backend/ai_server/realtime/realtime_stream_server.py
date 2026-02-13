"""
Real-Time Stream Server
WebSocket server that streams confidence data to frontend
Why: Provides continuous heartbeat-style data for UI visualization
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import time
import threading
from typing import Dict, List
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime.live_audio_capture import LiveAudioCapture
from realtime.live_feature_extractor import LiveFeatureExtractor
from realtime.real_time_inference import RealTimeInference
from realtime.confidence_smoother import ConfidenceSmoother

app = FastAPI(title="Real-Time Voice Confidence API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
audio_capture = None
feature_extractor = None
inference_engine = None
smoother = None

# Active WebSocket connections
active_connections: List[WebSocket] = []

# Configuration
CONFIG = {
    'sample_rate': 22050,
    'window_duration': 2.0,
    'overlap': 0.5,
    'update_interval': 0.3,  # 300ms updates for smooth heartbeat
    'model_path': '../../../models/trained/voice_stress_model.pth',
    'scaler_path': '../../../datasets/ravdess/preprocessed/scaler.pkl'
}


def initialize_components():
    """
    Initialize all components
    Why: Sets up the complete pipeline
    """
    global audio_capture, feature_extractor, inference_engine, smoother
    
    print("Initializing real-time voice confidence system...")
    
    # Audio capture
    audio_capture = LiveAudioCapture(
        sample_rate=CONFIG['sample_rate'],
        window_duration=CONFIG['window_duration'],
        overlap=CONFIG['overlap']
    )
    
    # Feature extractor
    feature_extractor = LiveFeatureExtractor(
        sample_rate=CONFIG['sample_rate'],
        scaler_path=CONFIG['scaler_path']
    )
    
    # Inference engine
    inference_engine = RealTimeInference(
        model_path=CONFIG['model_path'],
        device='cpu'
    )
    
    # Smoother
    smoother = ConfidenceSmoother(
        window_size=5,
        method='ema',
        alpha=0.3
    )
    
    print("All components initialized successfully")


async def process_audio_stream():
    """
    Main processing loop
    Why: Continuously processes audio and generates confidence data
    """
    while True:
        try:
            # Get audio window
            audio = audio_capture.get_audio_window()
            
            if audio is not None:
                # Extract features
                features = feature_extractor.extract_features(audio)
                
                if features is not None:
                    # Inference
                    result = inference_engine.predict(features)
                    
                    if result:
                        # Smooth confidence
                        smoothed_confidence = smoother.smooth(
                            result['confidence'],
                            result['stress_class']
                        )
                        
                        # Prepare data for frontend
                        data = {
                            'timestamp': int(time.time() * 1000),  # milliseconds
                            'confidence': round(smoothed_confidence, 2),
                            'stress_level': result['stress_level'],
                            'stress_class': result['stress_class'],
                            'probabilities': result['probabilities'],
                            'raw_confidence': round(result['confidence'], 2)
                        }
                        
                        # Broadcast to all connected clients
                        await broadcast_data(data)
            
            # Wait before next update (heartbeat interval)
            await asyncio.sleep(CONFIG['update_interval'])
            
        except Exception as e:
            print(f"Processing error: {e}")
            await asyncio.sleep(1)


async def broadcast_data(data: dict):
    """
    Broadcast data to all connected WebSocket clients
    Why: Sends confidence updates to all active UI connections
    """
    disconnected = []
    
    for connection in active_connections:
        try:
            await connection.send_json(data)
        except Exception as e:
            print(f"Error sending to client: {e}")
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    initialize_components()
    
    # Start audio capture
    audio_capture.start_capture()
    
    # Start processing loop
    asyncio.create_task(process_audio_stream())
    
    print("Real-time voice confidence server started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if audio_capture:
        audio_capture.stop_capture()
    print("Server shutdown")


@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "Real-Time Voice Confidence Analysis",
        "status": "running",
        "active_connections": len(active_connections)
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "audio_capture": audio_capture is not None,
            "feature_extractor": feature_extractor is not None,
            "inference_engine": inference_engine is not None,
            "smoother": smoother is not None
        },
        "config": CONFIG,
        "active_connections": len(active_connections)
    }


@app.websocket("/ws/voice-confidence")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time confidence streaming
    Why: Provides continuous data stream for heartbeat visualization
    
    Client receives:
    {
        "timestamp": 1700000000,
        "confidence": 72.4,
        "stress_level": "Low Stress",
        "stress_class": 0,
        "probabilities": [0.8, 0.15, 0.05],
        "raw_confidence": 75.2
    }
    """
    await websocket.accept()
    active_connections.append(websocket)
    
    print(f"Client connected. Total connections: {len(active_connections)}")
    
    try:
        # Send initial message
        await websocket.send_json({
            "type": "connected",
            "message": "Real-time voice confidence stream active",
            "update_interval_ms": CONFIG['update_interval'] * 1000
        })
        
        # Keep connection alive
        while True:
            # Wait for client messages (ping/pong)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                
                # Handle client commands
                if data == "reset":
                    smoother.reset()
                    await websocket.send_json({"type": "reset", "message": "Smoother reset"})
                    
            except asyncio.TimeoutError:
                # No message received, continue
                pass
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        print(f"Connection closed. Remaining: {len(active_connections)}")


@app.post("/control/start")
async def start_capture():
    """Start audio capture"""
    if audio_capture:
        audio_capture.start_capture()
        return {"status": "started"}
    return {"status": "error", "message": "Audio capture not initialized"}


@app.post("/control/stop")
async def stop_capture():
    """Stop audio capture"""
    if audio_capture:
        audio_capture.stop_capture()
        return {"status": "stopped"}
    return {"status": "error", "message": "Audio capture not initialized"}


@app.post("/control/reset")
async def reset_smoother():
    """Reset confidence smoother"""
    if smoother:
        smoother.reset()
        return {"status": "reset"}
    return {"status": "error", "message": "Smoother not initialized"}


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Real-Time Voice Confidence Server...")
    print(f"WebSocket endpoint: ws://localhost:8002/ws/voice-confidence")
    print(f"Update interval: {CONFIG['update_interval']*1000}ms")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
