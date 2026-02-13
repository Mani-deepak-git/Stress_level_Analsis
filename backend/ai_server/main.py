from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import asyncio
import json
import base64
import numpy as np
import cv2
import io
from PIL import Image
import wave
import struct
from typing import Dict, List
import logging
from inference_engine import create_analyzer, analyze_multimodal
from session_manager import session_manager
from pdf_generator import pdf_generator
from speech_analyzer import speech_analyzer
from alert_system import alert_system
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Interview Stress Analysis API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer instance
stress_analyzer = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict):
        disconnected_clients = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize the stress analyzer on startup"""
    global stress_analyzer
    logger.info("Initializing AI models...")
    
    try:
        stress_analyzer = create_analyzer()
        if stress_analyzer:
            logger.info("AI models loaded successfully!")
        else:
            logger.error("Failed to load AI models")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Interview Stress Analysis API",
        "status": "running",
        "models_loaded": stress_analyzer is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": stress_analyzer is not None,
        "active_connections": len(manager.active_connections)
    }

@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze single image for stress indicators"""
    if stress_analyzer is None:
        raise HTTPException(status_code=503, detail="AI models not loaded")
    
    try:
        # Read and decode image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Analyze frame
        results = stress_analyzer.analyze_frame(frame)
        
        return {
            "success": True,
            "results": results,
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/reset")
async def reset_analyzer():
    """Reset analyzer history"""
    if stress_analyzer is None:
        raise HTTPException(status_code=503, detail="AI models not loaded")
    
    try:
        stress_analyzer.reset_history()
        return {"success": True, "message": "Analyzer history reset"}
    except Exception as e:
        logger.error(f"Error resetting analyzer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time analysis"""
    logger.info(f"WebSocket connection attempt from client: {client_id}")
    await manager.connect(websocket, client_id)
    logger.info(f"WebSocket connected: {client_id}")
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            logger.info(f"Received message type: {message['type']} from {client_id}")
            
            if message["type"] == "video_frame":
                await handle_video_frame(message, client_id)
            elif message["type"] == "audio_chunk":
                await handle_audio_chunk(message, client_id)
            elif message["type"] == "multimodal_data":
                await handle_multimodal_data(message, client_id)
            elif message["type"] == "voice_confidence":
                await handle_voice_confidence(message, client_id)
            elif message["type"] == "reset":
                await handle_reset(client_id)
            else:
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"Unknown message type: {message['type']}"
                }, client_id)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client_id}")
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(client_id)

async def handle_video_frame(message: dict, client_id: str):
    """Handle video frame analysis"""
    logger.info(f"handle_video_frame called for client {client_id}")
    
    if stress_analyzer is None:
        await manager.send_personal_message({
            "type": "error",
            "message": "AI models not loaded"
        }, client_id)
        return
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(message["data"])
        image = Image.open(io.BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Get audio from buffer if available
        audio_chunk = list(stress_analyzer.audio_buffer) if len(stress_analyzer.audio_buffer) > 0 else None
        
        logger.info(f"Calling analyze_frame with audio_chunk: {audio_chunk is not None}, buffer size: {len(stress_analyzer.audio_buffer)}")
        
        # Analyze frame with audio
        results = stress_analyzer.analyze_frame(frame, audio_chunk)
        
        logger.info(f"Analysis results: {results}")
        
        # Get session and add data
        session_id = message.get("session_id")
        if session_id:
            session = session_manager.get_session(session_id)
            if session:
                session.add_stress_data(results)
                
                # Check for alerts
                alerts = alert_system.check_stress_alert(
                    results.get('stress_level', 'Low Stress'),
                    results.get('confidence_score', 0.5),
                    results.get('face_detected', False)
                )
                
                # Add alerts to session and send to client
                for alert in alerts:
                    session.add_alert(alert['type'], alert['message'])
                    await manager.send_personal_message({
                        "type": "alert",
                        "data": alert
                    }, client_id)
                
                # Speech analysis if audio available
                if audio_chunk and len(audio_chunk) > 1000:
                    audio_array = np.array(audio_chunk, dtype=np.float32)
                    speech_metrics = speech_analyzer.analyze_audio(audio_array)
                    if speech_metrics:
                        session.add_speech_metric(speech_metrics)
                        
                        # Check speech alerts
                        speech_alerts = alert_system.check_speech_alert(speech_metrics)
                        for alert in speech_alerts:
                            session.add_alert(alert['type'], alert['message'])
                            await manager.send_personal_message({
                                "type": "alert",
                                "data": alert
                            }, client_id)
                        
                        # Send speech metrics
                        await manager.send_personal_message({
                            "type": "speech_metrics",
                            "data": speech_metrics
                        }, client_id)
        
        # Send results back
        await manager.send_personal_message({
            "type": "analysis_result",
            "data": results,
            "timestamp": message.get("timestamp")
        }, client_id)
        
    except Exception as e:
        logger.error(f"Error handling video frame: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": str(e)
        }, client_id)

async def handle_audio_chunk(message: dict, client_id: str):
    """Handle audio chunk processing"""
    if stress_analyzer is None:
        await manager.send_personal_message({
            "type": "error",
            "message": "AI models not loaded"
        }, client_id)
        return
    
    try:
        # Decode audio data
        audio_data = message["data"]  # Audio samples
        sample_rate = message.get("sampleRate", 44100)
        
        logger.info(f"Received audio chunk: {len(audio_data)} samples at {sample_rate}Hz")
        
        # Add to analyzer's audio buffer
        stress_analyzer.audio_buffer.extend(audio_data)
        
        logger.info(f"Audio buffer size: {len(stress_analyzer.audio_buffer)}")
        
        await manager.send_personal_message({
            "type": "audio_received",
            "message": f"Audio chunk processed: {len(audio_data)} samples",
            "buffer_size": len(stress_analyzer.audio_buffer)
        }, client_id)
        
    except Exception as e:
        logger.error(f"Error handling audio chunk: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": str(e)
        }, client_id)

async def handle_multimodal_data(message: dict, client_id: str):
    """Handle combined video and audio data"""
    if stress_analyzer is None:
        await manager.send_personal_message({
            "type": "error",
            "message": "AI models not loaded"
        }, client_id)
        return
    
    try:
        # Decode video frame
        image_data = base64.b64decode(message["video_data"])
        image = Image.open(io.BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Get audio data
        audio_chunk = message.get("audio_data", None)
        
        # Analyze multimodal data
        results = analyze_multimodal(stress_analyzer, frame, audio_chunk)
        
        # Send results back
        await manager.send_personal_message({
            "type": "multimodal_result",
            "data": results,
            "timestamp": message.get("timestamp")
        }, client_id)
        
    except Exception as e:
        logger.error(f"Error handling multimodal data: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": str(e)
        }, client_id)

async def handle_reset(client_id: str):
    """Handle analyzer reset request"""
    if stress_analyzer is None:
        await manager.send_personal_message({
            "type": "error",
            "message": "AI models not loaded"
        }, client_id)
        return
    
    try:
        stress_analyzer.reset_history()
        await manager.send_personal_message({
            "type": "reset_complete",
            "message": "Analyzer history reset successfully"
        }, client_id)
        
    except Exception as e:
        logger.error(f"Error resetting analyzer: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": str(e)
        }, client_id)

async def handle_voice_confidence(message: dict, client_id: str):
    """Handle voice confidence data from frontend"""
    try:
        session_id = message.get("session_id")
        voice_data = message.get("data", {})
        
        if session_id:
            session = session_manager.get_session(session_id)
            if session:
                session.add_voice_confidence(voice_data)
                logger.info(f"Added voice confidence to session {session_id}: {voice_data.get('confidence', 0):.1f}%")
    except Exception as e:
        logger.error(f"Error handling voice confidence: {e}")

@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    return {
        "active_connections": len(manager.active_connections),
        "connected_clients": list(manager.active_connections.keys()),
        "models_loaded": stress_analyzer is not None
    }

# Session Management Endpoints
@app.post("/session/start")
async def start_session(data: dict):
    """Start new interview session"""
    session = session_manager.create_session(
        data['session_id'],
        data['interviewer'],
        data['interviewee']
    )
    alert_system.reset()
    speech_analyzer.reset()
    return {"success": True, "session_id": session.session_id}

@app.post("/session/end")
async def end_session(data: dict):
    """End interview session and get summary"""
    summary = session_manager.end_session(data['session_id'])
    if summary:
        return {"success": True, "summary": summary}
    return {"success": False, "message": "Session not found"}

@app.get("/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """Get session summary"""
    session = session_manager.get_completed_session(session_id)
    if session:
        return {"success": True, "summary": session.get_summary()}
    return {"success": False, "message": "Session not found"}

@app.post("/session/{session_id}/export-pdf")
async def export_session_pdf(session_id: str):
    """Export session as PDF report"""
    session = session_manager.get_completed_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        pdf_path = pdf_generator.generate_report(session.get_full_data())
        return FileResponse(
            pdf_path,
            media_type='application/pdf',
            filename=f"interview_report_{session_id}.pdf"
        )
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/alerts")
async def get_session_alerts(session_id: str):
    """Get all alerts for a session"""
    session = session_manager.get_session(session_id) or session_manager.get_completed_session(session_id)
    if session:
        return {"success": True, "alerts": session.alerts}
    return {"success": False, "message": "Session not found"}

# Additional utility endpoints
@app.post("/test/dummy_analysis")
async def test_dummy_analysis():
    """Test endpoint with dummy data"""
    if stress_analyzer is None:
        raise HTTPException(status_code=503, detail="AI models not loaded")
    
    try:
        # Create dummy data
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_audio = np.random.randn(8000).tolist()  # 0.5 seconds
        
        # Analyze
        results = analyze_multimodal(stress_analyzer, dummy_frame, dummy_audio)
        
        return {
            "success": True,
            "message": "Dummy analysis completed",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in dummy analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )