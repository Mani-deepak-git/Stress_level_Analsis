import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './InterviewSummary.css';

const InterviewSummary = ({ sessionId, onClose }) => {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    fetchSummary();
  }, [sessionId]);

  const fetchSummary = async () => {
    try {
      const response = await axios.get(`http://localhost:3000/api/session/${sessionId}/summary`);
      if (response.data.success) {
        setSummary(response.data.summary);
      }
    } catch (error) {
      console.error('Error fetching summary:', error);
    } finally {
      setLoading(false);
    }
  };

  const downloadPDF = async () => {
    setDownloading(true);
    try {
      const response = await axios.get(
        `http://localhost:3000/api/session/${sessionId}/export-pdf`,
        { responseType: 'blob' }
      );
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `interview_report_${sessionId}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Error downloading PDF:', error);
      alert('Failed to download PDF report');
    } finally {
      setDownloading(false);
    }
  };

  if (loading) {
    return (
      <div className="summary-modal">
        <div className="summary-content">
          <div className="loading-spinner"></div>
          <p>Loading summary...</p>
        </div>
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="summary-modal">
        <div className="summary-content">
          <p>No summary available</p>
          <button onClick={onClose}>Close</button>
        </div>
      </div>
    );
  }

  const stats = summary.stats;
  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  return (
    <div className="summary-modal" onClick={onClose}>
      <div className="summary-content" onClick={(e) => e.stopPropagation()}>
        <div className="summary-header">
          <h2>📊 Interview Summary</h2>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        <div className="summary-body">
          <div className="summary-section">
            <h3>Session Information</h3>
            <div className="info-grid">
              <div className="info-item">
                <span className="info-label">Interviewer:</span>
                <span className="info-value">{summary.interviewer}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Interviewee:</span>
                <span className="info-value">{summary.interviewee}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Duration:</span>
                <span className="info-value">{formatDuration(stats.total_duration)}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Data Points:</span>
                <span className="info-value">{summary.total_data_points}</span>
              </div>
            </div>
          </div>

          <div className="summary-section">
            <h3>Key Metrics</h3>
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-icon">🧠</div>
                <div className="metric-label">Avg Stress</div>
                <div className="metric-value">{stats.avg_stress.toFixed(2)}</div>
                <div className="metric-desc">
                  {stats.avg_stress < 1.3 ? 'Excellent' : stats.avg_stress < 1.6 ? 'Good' : 'High'}
                </div>
              </div>
              
              <div className="metric-card">
                <div className="metric-icon">🎯</div>
                <div className="metric-label">Avg Confidence</div>
                <div className="metric-value">{(stats.avg_confidence * 100).toFixed(1)}%</div>
                <div className="metric-desc">
                  {stats.avg_confidence > 0.7 ? 'High' : stats.avg_confidence > 0.4 ? 'Moderate' : 'Low'}
                </div>
              </div>
              
              <div className="metric-card">
                <div className="metric-icon">🎤</div>
                <div className="metric-label">Voice Confidence</div>
                <div className="metric-value">{stats.avg_voice_confidence.toFixed(1)}%</div>
                <div className="metric-desc">
                  {stats.avg_voice_confidence > 70 ? 'Strong' : stats.avg_voice_confidence > 40 ? 'Moderate' : 'Weak'}
                </div>
              </div>
              
              <div className="metric-card">
                <div className="metric-icon">⚠️</div>
                <div className="metric-label">Total Alerts</div>
                <div className="metric-value">{stats.total_alerts}</div>
                <div className="metric-desc">
                  {stats.total_alerts === 0 ? 'Smooth' : stats.total_alerts < 5 ? 'Few concerns' : 'Multiple'}
                </div>
              </div>
            </div>
          </div>

          <div className="summary-section">
            <h3>Stress Distribution</h3>
            <div className="stress-bars">
              <div className="stress-bar-item">
                <span className="stress-bar-label">Low Stress</span>
                <div className="stress-bar-container">
                  <div 
                    className="stress-bar-fill low"
                    style={{ width: `${(stats.low_stress_duration / stats.total_duration) * 100}%` }}
                  ></div>
                </div>
                <span className="stress-bar-value">{formatDuration(stats.low_stress_duration)}</span>
              </div>
              <div className="stress-bar-item">
                <span className="stress-bar-label">High Stress</span>
                <div className="stress-bar-container">
                  <div 
                    className="stress-bar-fill high"
                    style={{ width: `${(stats.high_stress_duration / stats.total_duration) * 100}%` }}
                  ></div>
                </div>
                <span className="stress-bar-value">{formatDuration(stats.high_stress_duration)}</span>
              </div>
            </div>
          </div>

          {summary.alerts && summary.alerts.length > 0 && (
            <div className="summary-section">
              <h3>Alerts ({summary.alerts.length})</h3>
              <div className="alerts-list">
                {summary.alerts.slice(0, 5).map((alert, index) => (
                  <div key={index} className="alert-item">
                    <span className="alert-time">
                      {new Date(alert.timestamp * 1000).toLocaleTimeString()}
                    </span>
                    <span className="alert-msg">{alert.message}</span>
                  </div>
                ))}
                {summary.alerts.length > 5 && (
                  <div className="alert-more">+ {summary.alerts.length - 5} more alerts</div>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="summary-footer">
          <button className="btn-secondary" onClick={onClose}>Close</button>
          <button 
            className="btn-primary" 
            onClick={downloadPDF}
            disabled={downloading}
          >
            {downloading ? 'Downloading...' : '📄 Download PDF Report'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default InterviewSummary;
