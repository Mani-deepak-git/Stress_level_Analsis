import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './HomePage.css';

const HomePage = () => {
  const [roomId, setRoomId] = useState('');
  const [userName, setUserName] = useState('');
  const [role, setRole] = useState('interviewer');
  const navigate = useNavigate();

  const generateRoomId = () => {
    const id = Math.random().toString(36).substring(2, 15);
    setRoomId(id);
  };

  const joinRoom = () => {
    if (!roomId.trim() || !userName.trim()) {
      alert('Please enter both room ID and your name');
      return;
    }
    navigate(`/interview/${roomId}?role=${role}&name=${encodeURIComponent(userName)}`);
  };

  return (
    <div className="home-container">
      <div className="home-background">
        <div className="gradient-orb orb-1"></div>
        <div className="gradient-orb orb-2"></div>
        <div className="gradient-orb orb-3"></div>
      </div>

      <div className="home-content">
        <div className="logo-section">
          <div className="logo-icon">🎯</div>
          <h1>Interview Stress Analyzer</h1>
          <p className="subtitle">AI-Powered Real-Time Behavioral Analysis</p>
        </div>

        <div className="form-container">
          <div className="form-group">
            <label>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M12 12C14.21 12 16 10.21 16 8C16 5.79 14.21 4 12 4C9.79 4 8 5.79 8 8C8 10.21 9.79 12 12 12ZM12 14C9.33 14 4 15.34 4 18V20H20V18C20 15.34 14.67 14 12 14Z" fill="currentColor"/>
              </svg>
              Your Name
            </label>
            <input
              type="text"
              value={userName}
              onChange={(e) => setUserName(e.target.value)}
              placeholder="Enter your full name"
              className="input-field"
            />
          </div>

          <div className="form-group">
            <label>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2Z" fill="currentColor"/>
              </svg>
              Select Role
            </label>
            <div className="role-selector">
              <div 
                className={`role-option ${role === 'interviewer' ? 'active' : ''}`}
                onClick={() => setRole('interviewer')}
              >
                <div className="role-icon">👔</div>
                <div className="role-text">
                  <h4>Interviewer</h4>
                  <p>View AI analytics</p>
                </div>
              </div>
              <div 
                className={`role-option ${role === 'interviewee' ? 'active' : ''}`}
                onClick={() => setRole('interviewee')}
              >
                <div className="role-icon">👤</div>
                <div className="role-text">
                  <h4>Interviewee</h4>
                  <p>Join interview</p>
                </div>
              </div>
            </div>
          </div>

          <div className="form-group">
            <label>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3Z" fill="currentColor"/>
              </svg>
              Room Code
            </label>
            <div className="room-input-group">
              <input
                type="text"
                value={roomId}
                onChange={(e) => setRoomId(e.target.value)}
                placeholder={role === 'interviewer' ? "Generate or enter code" : "Enter room code"}
                className="input-field"
              />
              {role === 'interviewer' && (
                <button onClick={generateRoomId} className="generate-btn">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                    <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4C7.58 4 4.01 7.58 4.01 12C4.01 16.42 7.58 20 12 20C15.73 20 18.84 17.45 19.73 14H17.65C16.83 16.33 14.61 18 12 18C8.69 18 6 15.31 6 12C6 8.69 8.69 6 12 6C13.66 6 15.14 6.69 16.22 7.78L13 11H20V4L17.65 6.35Z" fill="currentColor"/>
                  </svg>
                  Generate
                </button>
              )}
            </div>
          </div>

          <button onClick={joinRoom} className="join-btn">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M19 19H5V5H19M19 3H5C3.89 3 3 3.89 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.89 20.1 3 19 3M11 7H13V11H17V13H13V17H11V13H7V11H11V7Z" fill="currentColor"/>
            </svg>
            Join Interview Room
          </button>
        </div>

        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">🎥</div>
            <h3>HD Video</h3>
            <p>Crystal clear calls</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🤖</div>
            <h3>AI Analysis</h3>
            <p>Real-time detection</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🔒</div>
            <h3>Secure</h3>
            <p>Privacy protected</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Insights</h3>
            <p>Live analytics</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;