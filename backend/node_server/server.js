const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const axios = require('axios');
const WebSocket = require('ws');
const { v4: uuidv4 } = require('uuid');

const app = express();
const server = http.createServer(app);

// CORS configuration
app.use(cors({
    origin: ["http://localhost:3001", "http://localhost:3000"],
    credentials: true
}));

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Socket.IO setup
const io = socketIo(server, {
    cors: {
        origin: ["http://localhost:3001", "http://localhost:3000"],
        methods: ["GET", "POST"],
        credentials: true
    },
    maxHttpBufferSize: 1e8 // 100MB for large video frames
});

// AI Backend connection
const AI_BACKEND_URL = 'http://localhost:8001';
let aiWebSocket = null;

// Room management
const rooms = new Map();
const userRoles = new Map();
const userNames = new Map(); // Store user names

// Connect to AI backend WebSocket
function connectToAIBackend() {
    try {
        aiWebSocket = new WebSocket(`ws://localhost:8001/ws/node_server`);
        
        aiWebSocket.on('open', () => {
            console.log('Connected to AI backend');
        });
        
        aiWebSocket.on('message', (data) => {
            try {
                const message = JSON.parse(data);
                handleAIResponse(message);
            } catch (error) {
                console.error('Error parsing AI response:', error);
            }
        });
        
        aiWebSocket.on('close', () => {
            console.log('AI backend connection closed. Reconnecting...');
            setTimeout(connectToAIBackend, 5000);
        });
        
        aiWebSocket.on('error', (error) => {
            console.error('AI backend WebSocket error:', error);
        });
        
    } catch (error) {
        console.error('Error connecting to AI backend:', error);
        setTimeout(connectToAIBackend, 5000);
    }
}

// Handle AI responses
function handleAIResponse(message) {
    if (message.type === 'multimodal_result' || message.type === 'analysis_result') {
        // Broadcast analysis results to interviewers only
        const analysisData = {
            type: 'stress_analysis',
            data: message.data,
            timestamp: message.timestamp || Date.now()
        };
        
        // Send to all interviewers in all rooms
        rooms.forEach((roomData, roomId) => {
            roomData.participants.forEach(socketId => {
                const role = userRoles.get(socketId);
                if (role === 'interviewer') {
                    io.to(socketId).emit('stress_analysis', analysisData);
                }
            });
        });
    } else if (message.type === 'alert') {
        // Forward alerts to interviewers
        rooms.forEach((roomData, roomId) => {
            roomData.participants.forEach(socketId => {
                const role = userRoles.get(socketId);
                if (role === 'interviewer') {
                    io.to(socketId).emit('real_time_alert', message.data);
                }
            });
        });
    } else if (message.type === 'speech_metrics') {
        // Forward speech metrics to interviewers
        rooms.forEach((roomData, roomId) => {
            roomData.participants.forEach(socketId => {
                const role = userRoles.get(socketId);
                if (role === 'interviewer') {
                    io.to(socketId).emit('speech_metrics', message.data);
                }
            });
        });
    }
}

// Initialize AI connection
connectToAIBackend();

// Routes
app.get('/', (req, res) => {
    res.json({
        message: 'Interview WebRTC Server',
        status: 'running',
        aiConnected: aiWebSocket && aiWebSocket.readyState === WebSocket.OPEN
    });
});

app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        activeRooms: rooms.size,
        aiBackendConnected: aiWebSocket && aiWebSocket.readyState === WebSocket.OPEN
    });
});

// Socket.IO connection handling
io.on('connection', (socket) => {
    console.log(`User connected: ${socket.id}`);
    
    // Join room
    socket.on('join-room', (data) => {
        const { roomId, role, userName } = data;
        
        // Store user role and name
        userRoles.set(socket.id, role);
        userNames.set(socket.id, userName);
        console.log(`Stored userName for ${socket.id}: ${userName}`);
        console.log(`Current userNames map:`, Array.from(userNames.entries()));
        
        // Join socket room
        socket.join(roomId);
        
        // Update room data
        if (!rooms.has(roomId)) {
            rooms.set(roomId, {
                participants: new Set(),
                interviewers: new Set(),
                interviewees: new Set()
            });
        }
        
        const room = rooms.get(roomId);
        room.participants.add(socket.id);
        
        if (role === 'interviewer') {
            room.interviewers.add(socket.id);
        } else if (role === 'interviewee') {
            room.interviewees.add(socket.id);
        }
        
        console.log(`${userName} (${role}) joined room ${roomId}`);
        
        // Notify others in the room
        socket.to(roomId).emit('user-joined', {
            socketId: socket.id,
            role: role,
            userName: userName
        });
        
        // Send current room participants to the new user
        const participants = Array.from(room.participants)
            .filter(id => id !== socket.id)
            .map(id => ({
                socketId: id,
                role: userRoles.get(id)
            }));
        
        socket.emit('room-participants', participants);
    });
    
    // WebRTC signaling
    socket.on('offer', (data) => {
        socket.to(data.target).emit('offer', {
            offer: data.offer,
            sender: socket.id
        });
    });
    
    socket.on('answer', (data) => {
        socket.to(data.target).emit('answer', {
            answer: data.answer,
            sender: socket.id
        });
    });
    
    socket.on('ice-candidate', (data) => {
        socket.to(data.target).emit('ice-candidate', {
            candidate: data.candidate,
            sender: socket.id
        });
    });
    
    // AI Analysis - Video frames
    socket.on('video-frame', (data) => {
        const role = userRoles.get(socket.id);
        const roomId = Array.from(socket.rooms).find(room => room !== socket.id);
        
        // Only process interviewee's video
        if (role === 'interviewee' && aiWebSocket && aiWebSocket.readyState === WebSocket.OPEN) {
            const message = {
                type: 'video_frame',
                data: data.frame,
                timestamp: data.timestamp || Date.now(),
                userId: socket.id,
                session_id: roomId
            };
            
            aiWebSocket.send(JSON.stringify(message));
        }
    });
    
    // AI Analysis - Audio chunks
    socket.on('audio-chunk', (data) => {
        const role = userRoles.get(socket.id);
        
        console.log(`Received audio chunk from ${role}: ${data.audioData ? data.audioData.length : 0} samples`);
        
        // Only process interviewee's audio
        if (role === 'interviewee' && aiWebSocket && aiWebSocket.readyState === WebSocket.OPEN) {
            const message = {
                type: 'audio_chunk',
                data: data.audioData,
                timestamp: data.timestamp || Date.now(),
                sampleRate: data.sampleRate || 44100,
                userId: socket.id
            };
            
            console.log(`Forwarding audio to AI backend: ${data.audioData.length} samples`);
            aiWebSocket.send(JSON.stringify(message));
        }
    });
    
    // AI Analysis - Combined multimodal data
    socket.on('multimodal-data', (data) => {
        const role = userRoles.get(socket.id);
        
        // Only process interviewee's data
        if (role === 'interviewee' && aiWebSocket && aiWebSocket.readyState === WebSocket.OPEN) {
            const message = {
                type: 'multimodal_data',
                video_data: data.videoFrame,
                audio_data: data.audioChunk,
                timestamp: data.timestamp || Date.now(),
                userId: socket.id
            };
            
            aiWebSocket.send(JSON.stringify(message));
        }
    });
    
    // Reset AI analysis
    socket.on('reset-analysis', () => {
        if (aiWebSocket && aiWebSocket.readyState === WebSocket.OPEN) {
            const message = {
                type: 'reset',
                userId: socket.id
            };
            
            aiWebSocket.send(JSON.stringify(message));
        }
    });
    
    socket.on('chat-message', (data) => {
        const roomId = Array.from(socket.rooms).find(room => room !== socket.id);
        const senderName = userNames.get(socket.id) || 'Unknown';
        console.log(`Chat message from ${socket.id}: senderName=${senderName}, data.userName=${data.userName}`);
        if (roomId) {
            socket.to(roomId).emit('chat-message', {
                message: data.message,
                sender: senderName,
                timestamp: Date.now()
            });
            console.log(`Sent chat message to room ${roomId}: sender=${senderName}`);
        }
    });
    
    // Voice confidence data forwarding
    socket.on('voice-confidence-data', (data) => {
        const roomId = Array.from(socket.rooms).find(room => room !== socket.id);
        
        // Forward to AI backend for session tracking
        if (aiWebSocket && aiWebSocket.readyState === WebSocket.OPEN) {
            const message = {
                type: 'voice_confidence',
                data: data,
                session_id: roomId,
                timestamp: data.timestamp || Date.now()
            };
            aiWebSocket.send(JSON.stringify(message));
        }
    });
    
    // Handle disconnection
    socket.on('disconnect', () => {
        console.log(`User disconnected: ${socket.id}`);
        
        // Clean up room data
        rooms.forEach((roomData, roomId) => {
            if (roomData.participants.has(socket.id)) {
                roomData.participants.delete(socket.id);
                roomData.interviewers.delete(socket.id);
                roomData.interviewees.delete(socket.id);
                
                // Notify others in the room
                socket.to(roomId).emit('user-left', {
                    socketId: socket.id
                });
                
                // Remove empty rooms
                if (roomData.participants.size === 0) {
                    rooms.delete(roomId);
                }
            }
        });
        
        // Clean up user role and name
        userRoles.delete(socket.id);
        userNames.delete(socket.id);
    });
});

// API endpoints for room management
app.get('/api/rooms', (req, res) => {
    const roomList = Array.from(rooms.entries()).map(([roomId, data]) => ({
        roomId,
        participantCount: data.participants.size,
        interviewerCount: data.interviewers.size,
        intervieweeCount: data.interviewees.size
    }));
    
    res.json(roomList);
});

app.post('/api/rooms/create', (req, res) => {
    const roomId = uuidv4();
    res.json({ roomId });
});

// Test AI connection endpoint
app.get('/api/ai/test', async (req, res) => {
    try {
        const response = await axios.get(`${AI_BACKEND_URL}/health`);
        res.json({
            aiBackendStatus: 'connected',
            aiBackendData: response.data
        });
    } catch (error) {
        res.status(503).json({
            aiBackendStatus: 'disconnected',
            error: error.message
        });
    }
});

// Session management endpoints
app.post('/api/session/start', async (req, res) => {
    try {
        const response = await axios.post(`${AI_BACKEND_URL}/session/start`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/session/end', async (req, res) => {
    try {
        const response = await axios.post(`${AI_BACKEND_URL}/session/end`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/session/:sessionId/summary', async (req, res) => {
    try {
        const response = await axios.get(`${AI_BACKEND_URL}/session/${req.params.sessionId}/summary`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/session/:sessionId/export-pdf', async (req, res) => {
    try {
        const response = await axios.post(
            `${AI_BACKEND_URL}/session/${req.params.sessionId}/export-pdf`,
            {},
            { responseType: 'arraybuffer' }
        );
        res.setHeader('Content-Type', 'application/pdf');
        res.setHeader('Content-Disposition', `attachment; filename=interview_report_${req.params.sessionId}.pdf`);
        res.send(response.data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Error handling
app.use((error, req, res, next) => {
    console.error('Server error:', error);
    res.status(500).json({ error: 'Internal server error' });
});

const PORT = process.env.PORT || 3000;

server.listen(PORT, () => {
    console.log(`WebRTC server running on port ${PORT}`);
    console.log(`Health check: http://localhost:${PORT}/health`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('SIGTERM received, shutting down gracefully');
    server.close(() => {
        console.log('Server closed');
        if (aiWebSocket) {
            aiWebSocket.close();
        }
        process.exit(0);
    });
});