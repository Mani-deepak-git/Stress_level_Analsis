import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './StressAnalytics.css';

const StressAnalytics = ({ stressData, onReset }) => {
  const [historicalData, setHistoricalData] = useState([]);
  const [currentAnalysis, setCurrentAnalysis] = useState(null);
  const [voiceConfidence, setVoiceConfidence] = useState(null);
  const [voiceHistory, setVoiceHistory] = useState([]);
  const [wsConnected, setWsConnected] = useState(false);

  // WebSocket for voice confidence
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8002/ws/voice-confidence');
    
    ws.onopen = () => {
      setWsConnected(true);
      console.log('Voice confidence WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.confidence !== undefined) {
        setVoiceConfidence(data);
        
        // Send voice confidence to backend via socket for session tracking
        if (window.interviewSocket) {
          window.interviewSocket.emit('voice-confidence-data', {
            confidence: data.confidence,
            stress_level: data.stress_level,
            timestamp: data.timestamp
          });
        }
        
        setVoiceHistory(prev => {
          const newPoint = {
            time: new Date(data.timestamp).toLocaleTimeString(),
            confidence: data.confidence,
            timestamp: data.timestamp
          };
          return [...prev.slice(-20), newPoint];
        });
      }
    };
    
    ws.onerror = () => setWsConnected(false);
    ws.onclose = () => setWsConnected(false);
    
    return () => ws.close();
  }, []);

  useEffect(() => {
    if (stressData) {
      setCurrentAnalysis(stressData);
      
      const timestamp = new Date().toLocaleTimeString();
      const stressLevel = getStressLevelValue(stressData.stress_level);
      
      const newDataPoint = {
        time: timestamp,
        stressLevel: stressLevel,
        confidence: Math.round(stressData.confidence_score * 100),
        timestamp: Date.now()
      };

      setHistoricalData(prev => {
        const updated = [...prev, newDataPoint];
        return updated.slice(-20);
      });
    }
  }, [stressData]);

  const getStressLevelValue = (level) => {
    if (level === 'Low Stress' || level === 'Medium Stress') return 1;
    if (level === 'High Stress') return 2;
    return 0;
  };

  const getStressColor = (level) => {
    if (level === 'Low Stress' || level === 'Medium Stress') return 'linear-gradient(135deg, #00d2ff, #3a7bd5)';
    if (level === 'High Stress') return 'linear-gradient(135deg, #ef4444, #dc2626)';
    return '#9E9E9E';
  };

  const getConfidenceColor = (score) => {
    if (score >= 0.6) return 'linear-gradient(135deg, #00d2ff, #3a7bd5)';
    return 'linear-gradient(135deg, #ef4444, #dc2626)';
  };

  const clearHistory = () => {
    setHistoricalData([]);
    setCurrentAnalysis(null);
    if (onReset) onReset();
  };

  const displayStressLevel = (level) => {
    if (level === 'Medium Stress') return 'Low Stress';
    return level;
  };

  if (!currentAnalysis) {
    return (
      <div className="stress-analytics">
        <div className="analytics-header">
          <div className="header-title">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM13 17H11V15H13V17ZM13 13H11V7H13V13Z" fill="currentColor"/>
            </svg>
            <h3>AI Stress Analysis</h3>
          </div>
          <button onClick={clearHistory} className="reset-btn">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4C7.58 4 4.01 7.58 4.01 12C4.01 16.42 7.58 20 12 20C15.73 20 18.84 17.45 19.73 14H17.65C16.83 16.33 14.61 18 12 18C8.69 18 6 15.31 6 12C6 8.69 8.69 6 12 6C13.66 6 15.14 6.69 16.22 7.78L13 11H20V4L17.65 6.35Z" fill="currentColor"/>
            </svg>
            Reset
          </button>
        </div>
        <div className="no-data">
          <div className="loading-icon">🧠</div>
          <p>Waiting for analysis data...</p>
          <div className="loading-spinner"></div>
        </div>
      </div>
    );
  }

  const displayLevel = displayStressLevel(currentAnalysis.stress_level);

  return (
    <div className="stress-analytics">
      <div className="analytics-header">
        <div className="header-title">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2Z" fill="currentColor"/>
          </svg>
          <h3>AI Analysis</h3>
        </div>
        <button onClick={clearHistory} className="reset-btn">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4C7.58 4 4.01 7.58 4.01 12C4.01 16.42 7.58 20 12 20C15.73 20 18.84 17.45 19.73 14H17.65C16.83 16.33 14.61 18 12 18C8.69 18 6 15.31 6 12C6 8.69 8.69 6 12 6C13.66 6 15.14 6.69 16.22 7.78L13 11H20V4L17.65 6.35Z" fill="currentColor"/>
          </svg>
          Reset
        </button>
      </div>

      <div className="current-status">
        <div className="status-card stress-card">
          <div className="card-icon">🧠</div>
          <div className="card-content">
            <h4>Stress Level</h4>
            <div 
              className="stress-indicator"
              style={{ background: getStressColor(currentAnalysis.stress_level) }}
            >
              {displayLevel}
            </div>
          </div>
        </div>

        {voiceConfidence && (
        <div className="status-card voice-card">
          <div className="card-icon">🎤</div>
          <div className="card-content">
            <h4>Voice Confidence</h4>
            <div 
              className="confidence-indicator"
              style={{ background: voiceConfidence.confidence >= 60 ? 'linear-gradient(135deg, #00d2ff, #3a7bd5)' : 'linear-gradient(135deg, #ef4444, #dc2626)' }}
            >
              {Math.round(voiceConfidence.confidence)}%
            </div>
          </div>
        </div>
      )}

      <div className="status-card confidence-card">
          <div className="card-icon">🎯</div>
          <div className="card-content">
            <h4>Confidence</h4>
            <div 
              className="confidence-indicator"
              style={{ background: getConfidenceColor(currentAnalysis.confidence_score) }}
            >
              {Math.round(currentAnalysis.confidence_score * 100)}%
            </div>
          </div>
        </div>
      </div>

      <div className="probability-section">
        <h4>📊 Live Stress Detection</h4>
        <div className="probability-bars">
          {currentAnalysis.stress_probability && currentAnalysis.stress_probability.map((prob, index) => {
            if (index === 1) return null; // Skip Medium Stress
            const labels = ['Low Stress', 'High Stress'];
            const colors = ['#00d2ff', '#ef4444'];
            const displayIndex = index === 0 ? 0 : 1;
            
            return (
              <div key={index} className="probability-bar">
                <div className="bar-label">{labels[displayIndex]}</div>
                <div className="bar-container">
                  <div 
                    className="bar-fill"
                    style={{ 
                      width: `${prob * 100}%`,
                      background: `linear-gradient(135deg, ${colors[displayIndex]}, ${colors[displayIndex]}dd)`,
                      transition: 'width 0.5s ease'
                    }}
                  ></div>
                </div>
                <div className="bar-value">{Math.round(prob * 100)}%</div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="detection-card">
        <h4>🔍 Detection Status</h4>
        <div className="detection-grid">
          <div className={`detection-item ${currentAnalysis.face_detected ? 'active' : 'inactive'}`}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M12 12C14.21 12 16 10.21 16 8C16 5.79 14.21 4 12 4C9.79 4 8 5.79 8 8C8 10.21 9.79 12 12 12Z" fill="currentColor"/>
            </svg>
            <span>Face {currentAnalysis.face_detected ? 'Detected' : 'Not Found'}</span>
          </div>
          <div className={`detection-item ${currentAnalysis.audio_processed ? 'active' : 'inactive'}`}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M12 14C13.66 14 15 12.66 15 11V5C15 3.34 13.66 2 12 2C10.34 2 9 3.34 9 5V11C9 12.66 10.34 14 12 14Z" fill="currentColor"/>
            </svg>
            <span>Audio {currentAnalysis.audio_processed ? 'Processing' : 'Inactive'}</span>
          </div>
        </div>
      </div>

      {historicalData.length > 1 && (
        <div className="chart-card">
          <h4>📈 Stress Trend</h4>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={historicalData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="time" tick={{fontSize: 11}} stroke="#94a3b8" />
              <YAxis domain={[0, 2]} ticks={[1, 2]} tickFormatter={(value) => value === 1 ? 'Low' : 'High'} tick={{fontSize: 11}} stroke="#94a3b8" />
              <Tooltip formatter={(value) => [value === 1 ? 'Low Stress' : 'High Stress', 'Level']} />
              <Line type="monotone" dataKey="stressLevel" stroke="#00d2ff" strokeWidth={3} dot={{ fill: '#00d2ff', r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* {historicalData.length > 1 && (
        <div className="chart-card">
          <h4>💪 Confidence Trend</h4>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={historicalData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="time" tick={{fontSize: 11}} stroke="#94a3b8" />
              <YAxis domain={[0, 100]} tick={{fontSize: 11}} stroke="#94a3b8" />
              <Tooltip formatter={(value) => [`${value}%`, 'Confidence']} />
              <Line type="monotone" dataKey="confidence" stroke="#00d2ff" strokeWidth={3} dot={{ fill: '#00d2ff', r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )} */}

      {voiceHistory.length > 1 && (
        <div className="chart-card">
          <h4>🎤 Voice Confidence Trend</h4>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={voiceHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="time" tick={{fontSize: 11}} stroke="#94a3b8" />
              <YAxis domain={[0, 100]} tick={{fontSize: 11}} stroke="#94a3b8" />
              <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Voice Confidence']} />
              <Line type="monotone" dataKey="confidence" stroke="#3a7bd5" strokeWidth={3} dot={{ fill: '#3a7bd5', r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="recommendations-card">
        <h4>💡 Insights</h4>
        <div className="recommendation-list">
          {currentAnalysis.stress_level === 'High Stress' && (
            <div className="recommendation high">
              <span className="rec-icon">⚠️</span>
              <span>High stress detected. Consider a short break.</span>
            </div>
          )}
          {currentAnalysis.confidence_score < 0.3 && (
            <div className="recommendation low">
              <span className="rec-icon">💬</span>
              <span>Low confidence. Try encouraging questions.</span>
            </div>
          )}
          {!currentAnalysis.face_detected && (
            <div className="recommendation warning">
              <span className="rec-icon">📹</span>
              <span>Face not detected. Check camera position.</span>
            </div>
          )}
          {displayLevel === 'Low Stress' && currentAnalysis.confidence_score > 0.7 && (
            <div className="recommendation positive">
              <span className="rec-icon">✅</span>
              <span>Great! Candidate is comfortable and confident.</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StressAnalytics;