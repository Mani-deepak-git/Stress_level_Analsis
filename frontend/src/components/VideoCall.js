import React, { useEffect, useState, useRef } from 'react';
import './VideoCall.css';

const VideoCall = ({ 
  socket, 
  role, 
  participants, 
  localVideoRef, 
  remoteVideoRef, 
  peerConnectionRef, 
  localStreamRef 
}) => {
  const [isVideoEnabled, setIsVideoEnabled] = useState(true);
  const [isAudioEnabled, setIsAudioEnabled] = useState(true);
  const [isCallActive, setIsCallActive] = useState(false);
  
  const audioContextRef = useRef();
  const analyserRef = useRef();
  const dataArrayRef = useRef();
  const frameIntervalRef = useRef();
  const audioIntervalRef = useRef();

  // WebRTC configuration
  const rtcConfig = {
    iceServers: [
      { urls: 'stun:stun.l.google.com:19302' },
      { urls: 'stun:stun1.l.google.com:19302' }
    ]
  };

  useEffect(() => {
    if (socket) {
      setupSocketListeners();
      initializeMedia();
    }

    return () => {
      cleanup();
    };
  }, [socket]);

  const setupSocketListeners = () => {
    socket.on('offer', handleOffer);
    socket.on('answer', handleAnswer);
    socket.on('ice-candidate', handleIceCandidate);
  };

  const initializeMedia = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
      });

      localStreamRef.current = stream;
      if (localVideoRef.current) {
        localVideoRef.current.srcObject = stream;
      }

      // Setup audio analysis for interviewees
      if (role === 'interviewee') {
        setupAudioAnalysis(stream);
      }

      // Setup peer connection
      setupPeerConnection();

      // Start sending frames for AI analysis (interviewees only)
      if (role === 'interviewee') {
        startFrameCapture();
      }

    } catch (error) {
      console.error('Error accessing media devices:', error);
    }
  };

  const setupAudioAnalysis = (stream) => {
    try {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      
      analyserRef.current.fftSize = 2048;
      const bufferLength = analyserRef.current.fftSize; // Use fftSize for time domain data
      dataArrayRef.current = new Float32Array(bufferLength);
      
      source.connect(analyserRef.current);

      // Start audio chunk sending
      startAudioCapture();
    } catch (error) {
      console.error('Error setting up audio analysis:', error);
    }
  };

  const setupPeerConnection = () => {
    peerConnectionRef.current = new RTCPeerConnection(rtcConfig);

    // Add local stream tracks
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach(track => {
        peerConnectionRef.current.addTrack(track, localStreamRef.current);
      });
    }

    // Handle remote stream
    peerConnectionRef.current.ontrack = (event) => {
      if (remoteVideoRef.current) {
        remoteVideoRef.current.srcObject = event.streams[0];
        setIsCallActive(true);
      }
    };

    // Handle ICE candidates
    peerConnectionRef.current.onicecandidate = (event) => {
      if (event.candidate && socket) {
        // Send to all participants
        participants.forEach(participant => {
          socket.emit('ice-candidate', {
            candidate: event.candidate,
            target: participant.socketId
          });
        });
      }
    };
  };

  const startCall = async () => {
    if (!peerConnectionRef.current || participants.length === 0) return;

    try {
      // Create offer for each participant
      for (const participant of participants) {
        const offer = await peerConnectionRef.current.createOffer();
        await peerConnectionRef.current.setLocalDescription(offer);

        socket.emit('offer', {
          offer: offer,
          target: participant.socketId
        });
      }
    } catch (error) {
      console.error('Error starting call:', error);
    }
  };

  const handleOffer = async (data) => {
    try {
      if (!peerConnectionRef.current) {
        setupPeerConnection();
      }

      await peerConnectionRef.current.setRemoteDescription(data.offer);
      const answer = await peerConnectionRef.current.createAnswer();
      await peerConnectionRef.current.setLocalDescription(answer);

      socket.emit('answer', {
        answer: answer,
        target: data.sender
      });
    } catch (error) {
      console.error('Error handling offer:', error);
    }
  };

  const handleAnswer = async (data) => {
    try {
      await peerConnectionRef.current.setRemoteDescription(data.answer);
    } catch (error) {
      console.error('Error handling answer:', error);
    }
  };

  const handleIceCandidate = async (data) => {
    try {
      await peerConnectionRef.current.addIceCandidate(data.candidate);
    } catch (error) {
      console.error('Error handling ICE candidate:', error);
    }
  };

  const startFrameCapture = () => {
    frameIntervalRef.current = setInterval(() => {
      captureAndSendFrame();
    }, 2000); // Send frame every 2 seconds
  };

  const captureAndSendFrame = () => {
    if (!localVideoRef.current || !socket) return;

    try {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      canvas.width = 640;
      canvas.height = 480;
      
      ctx.drawImage(localVideoRef.current, 0, 0, canvas.width, canvas.height);
      
      // Convert to base64
      const frameData = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
      
      socket.emit('video-frame', {
        frame: frameData,
        timestamp: Date.now()
      });
    } catch (error) {
      console.error('Error capturing frame:', error);
    }
  };

  const startAudioCapture = () => {
    audioIntervalRef.current = setInterval(() => {
      captureAndSendAudio();
    }, 1000); // Send audio every 1 second
  };

  const captureAndSendAudio = () => {
    if (!analyserRef.current || !socket) return;

    try {
      // Get time domain data (actual audio samples)
      analyserRef.current.getFloatTimeDomainData(dataArrayRef.current);
      
      // Check if we have actual audio signal (not just silence)
      const hasSignal = dataArrayRef.current.some(sample => Math.abs(sample) > 0.01);
      
      if (!hasSignal) {
        console.log('No audio signal detected');
        return;
      }
      
      // Convert to regular array and normalize to 16-bit range
      const audioData = Array.from(dataArrayRef.current).map(sample => 
        Math.round(sample * 32767) // Convert from [-1, 1] to [-32767, 32767]
      );
      
      console.log(`Sending audio chunk: ${audioData.length} samples, max: ${Math.max(...audioData.map(Math.abs))}`);
      
      socket.emit('audio-chunk', {
        audioData: audioData,
        timestamp: Date.now(),
        sampleRate: audioContextRef.current.sampleRate
      });
    } catch (error) {
      console.error('Error capturing audio:', error);
    }
  };

  const toggleVideo = () => {
    if (localStreamRef.current) {
      const videoTrack = localStreamRef.current.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled;
        setIsVideoEnabled(videoTrack.enabled);
      }
    }
  };

  const toggleAudio = () => {
    if (localStreamRef.current) {
      const audioTrack = localStreamRef.current.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        setIsAudioEnabled(audioTrack.enabled);
      }
    }
  };

  const cleanup = () => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
    }
    if (audioIntervalRef.current) {
      clearInterval(audioIntervalRef.current);
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach(track => track.stop());
    }
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
    }
  };

  return (
    <div className="video-call-container">
      <div className="video-grid">
        <div className="video-item local-video">
          <video
            ref={localVideoRef}
            autoPlay
            muted
            playsInline
            className="video-element"
          />
          <div className="video-label">You ({role})</div>
        </div>

        <div className="video-item remote-video">
          <video
            ref={remoteVideoRef}
            autoPlay
            playsInline
            className="video-element"
          />
          <div className="video-label">
            {isCallActive ? 'Remote Participant' : 'Waiting for connection...'}
          </div>
        </div>
      </div>

      <div className="video-controls">
        <button
          onClick={toggleVideo}
          className={`control-btn ${isVideoEnabled ? 'active' : 'inactive'}`}
        >
          {isVideoEnabled ? '📹' : '📹❌'}
        </button>

        <button
          onClick={toggleAudio}
          className={`control-btn ${isAudioEnabled ? 'active' : 'inactive'}`}
        >
          {isAudioEnabled ? '🎤' : '🎤❌'}
        </button>

        <button
          onClick={startCall}
          className="control-btn call-btn"
          disabled={participants.length === 0}
        >
          📞 Call
        </button>
      </div>

      <div className="participants-list">
        <h4>Participants ({participants.length + 1})</h4>
        <div className="participant">You ({role})</div>
        {participants.map((participant, index) => (
          <div key={index} className="participant">
            {participant.role} (ID: {participant.socketId.substring(0, 8)})
          </div>
        ))}
      </div>
    </div>
  );
};

export default VideoCall;