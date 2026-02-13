import React, { useState, useEffect, useRef } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import io from 'socket.io-client';
import axios from 'axios';
import VideoCall from '../components/VideoCall';
import StressAnalytics from '../components/StressAnalytics';
import AlertContainer from '../components/AlertNotification';
import InterviewSummary from '../components/InterviewSummary';
import './InterviewRoom.css';

const InterviewRoom = () => {
  const { roomId } = useParams();
  const [searchParams] = useSearchParams();
  const role = searchParams.get('role');
  const userName = searchParams.get('name');

  const [socket, setSocket] = useState(null);
  const [participants, setParticipants] = useState([]);
  const [stressData, setStressData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [alerts, setAlerts] = useState([]);
  const [speechMetrics, setSpeechMetrics] = useState(null);
  const [showSummary, setShowSummary] = useState(false);
  const [sessionStarted, setSessionStarted] = useState(false);
  const navigate = useNavigate();

  const localVideoRef = useRef();
  const remoteVideoRef = useRef();
  const peerConnectionRef = useRef();
  const localStreamRef = useRef();

  useEffect(() => {
    const newSocket = io('http://localhost:3000');
    setSocket(newSocket);
    
    // Expose socket globally for voice confidence tracking
    window.interviewSocket = newSocket;

    newSocket.on('connect', () => {
      setIsConnected(true);
      newSocket.emit('join-room', { roomId, role, userName });
      
      // Start session if interviewer
      if (role === 'interviewer' && !sessionStarted) {
        startSession();
      }
    });

    newSocket.on('disconnect', () => {
      setIsConnected(false);
    });

    newSocket.on('user-joined', (data) => {
      setParticipants(prev => [...prev, data]);
    });

    newSocket.on('user-left', (data) => {
      setParticipants(prev => prev.filter(p => p.socketId !== data.socketId));
    });

    newSocket.on('room-participants', (data) => {
      setParticipants(data);
    });

    newSocket.on('stress_analysis', (data) => {
      if (role === 'interviewer') {
        setStressData(data.data);
      }
    });

    newSocket.on('real_time_alert', (alert) => {
      if (role === 'interviewer') {
        setAlerts(prev => [...prev, alert]);
      }
    });

    newSocket.on('speech_metrics', (metrics) => {
      if (role === 'interviewer') {
        setSpeechMetrics(metrics);
      }
    });

    newSocket.on('chat-message', (data) => {
      setChatMessages(prev => [...prev, data]);
    });

    return () => {
      newSocket.disconnect();
      window.interviewSocket = null;
    };
  }, [roomId, role, userName]);

  const startSession = async () => {
    try {
      await axios.post('http://localhost:3000/api/session/start', {
        session_id: roomId,
        interviewer: userName,
        interviewee: 'Candidate'
      });
      setSessionStarted(true);
    } catch (error) {
      console.error('Error starting session:', error);
    }
  };

  const endSession = async () => {
    try {
      await axios.post('http://localhost:3000/api/session/end', {
        session_id: roomId
      });
      setShowSummary(true);
    } catch (error) {
      console.error('Error ending session:', error);
    }
  };

  const dismissAlert = (index) => {
    setAlerts(prev => prev.filter((_, i) => i !== index));
  };

  const sendChatMessage = () => {
    if (newMessage.trim() && socket) {
      socket.emit('chat-message', { message: newMessage, userName: userName });
      setChatMessages(prev => [...prev, {
        message: newMessage,
        sender: userName,
        timestamp: Date.now()
      }]);
      setNewMessage('');
    }
  };

  const resetAnalysis = () => {
    if (socket) {
      socket.emit('reset-analysis');
      setStressData(null);
    }
  };

  return (
    <div className="interview-room">
      <div className="room-header">
        <div className="header-left">
          <div className="room-icon">🎯</div>
          <div>
            <h2>Room: {roomId}</h2>
            <span className="room-subtitle">{userName} • {role}</span>
          </div>
        </div>
        <div className="header-right">
          {role === 'interviewer' && (
            <button className="end-interview-btn" onClick={endSession}>
              📊 End & View Summary
            </button>
          )}
          <span className={`status-badge ${isConnected ? 'connected' : 'disconnected'}`}>
            <span className="status-dot"></span>
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      <div className="room-content">
        <div className="left-panel">
          <div className="chat-section">
            <div className="chat-header">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M20 2H4C2.9 2 2 2.9 2 4V22L6 18H20C21.1 18 22 17.1 22 16V4C22 2.9 21.1 2 20 2Z" fill="currentColor"/>
              </svg>
              <h3>Chat</h3>
            </div>
            <div className="chat-messages">
              {chatMessages.map((msg, index) => (
                <div key={index} className={`chat-message ${msg.sender === userName ? 'own' : 'other'}`}>
                  <div className="message-avatar">{msg.sender.charAt(0).toUpperCase()}</div>
                  <div className="message-content">
                    <div className="message-header">
                      <span className="sender-name">{msg.sender}</span>
                      <span className="message-time">{new Date(msg.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                    </div>
                    <div className="message-text">{msg.message}</div>
                  </div>
                </div>
              ))}
            </div>
            <div className="chat-input">
              <input
                type="text"
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
                placeholder="Type a message..."
              />
              <button onClick={sendChatMessage} className="send-btn">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="currentColor"/>
                </svg>
              </button>
            </div>
          </div>
        </div>

        <div className="center-panel">
          <VideoCall
            socket={socket}
            role={role}
            participants={participants}
            localVideoRef={localVideoRef}
            remoteVideoRef={remoteVideoRef}
            peerConnectionRef={peerConnectionRef}
            localStreamRef={localStreamRef}
          />
        </div>

        {role === 'interviewer' && (
          <div className="right-panel">
            <StressAnalytics 
              stressData={stressData}
              onReset={resetAnalysis}
            />
            {speechMetrics && (
              <div className="speech-metrics-card">
                <h4>🗣️ Speech Analysis</h4>
                <div className="speech-metric">
                  <span>Speaking Pace:</span>
                  <strong>{speechMetrics.speaking_pace} WPM</strong>
                </div>
                <div className="speech-metric">
                  <span>Pause Ratio:</span>
                  <strong>{(speechMetrics.pause_ratio * 100).toFixed(0)}%</strong>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {role === 'interviewer' && alerts.length > 0 && (
        <AlertContainer alerts={alerts} onDismiss={dismissAlert} />
      )}

      {showSummary && (
        <InterviewSummary 
          sessionId={roomId} 
          onClose={() => {
            setShowSummary(false);
            navigate('/');
          }} 
        />
      )}
    </div>
  );
};

export default InterviewRoom;