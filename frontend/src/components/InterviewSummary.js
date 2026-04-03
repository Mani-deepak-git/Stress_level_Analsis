import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './InterviewSummary.css';

// Build summary purely from frontend-collected data (works even when backend is sleeping)
function buildLocalSummary(sessionId, interviewerName, sessionStartTime, localStressData, localAlerts) {
  const now = Date.now();
  const totalDuration = (now - sessionStartTime) / 1000;

  let avgStress = 0, avgConfidence = 0, highCount = 0, lowCount = 0;
  if (localStressData.length > 0) {
    const stressVals = localStressData.map(d =>
      d.stress_level === 'High Stress' ? 2 : 1
    );
    avgStress = stressVals.reduce((a, b) => a + b, 0) / stressVals.length;
    avgConfidence = localStressData.reduce((a, b) => a + (b.confidence_score || 0), 0) / localStressData.length;
    highCount = stressVals.filter(v => v === 2).length;
    lowCount = stressVals.filter(v => v === 1).length;
  }

  const highStressDuration = localStressData.length > 0
    ? (highCount / localStressData.length) * totalDuration : 0;
  const lowStressDuration = totalDuration - highStressDuration;

  return {
    session_id: sessionId,
    interviewer: interviewerName || 'Interviewer',
    interviewee: 'Candidate',
    start_time: new Date(sessionStartTime).toISOString(),
    end_time: new Date(now).toISOString(),
    total_data_points: localStressData.length,
    total_voice_points: 0,
    alerts: localAlerts || [],
    stats: {
      total_duration: totalDuration,
      avg_stress: avgStress,
      avg_confidence: avgConfidence,
      avg_voice_confidence: 0,
      high_stress_duration: highStressDuration,
      low_stress_duration: lowStressDuration,
      total_alerts: (localAlerts || []).length
    }
  };
}

const InterviewSummary = ({ sessionId, onClose, sessionStartTime, interviewerName, localStressData, localAlerts }) => {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);
  const [dataSource, setDataSource] = useState('loading'); // 'server' | 'local'

  const fetchSummary = useCallback(async () => {
    // 1. Show local summary immediately — never block on the server
    const local = buildLocalSummary(
      sessionId, interviewerName, sessionStartTime,
      localStressData || [], localAlerts || []
    );
    setSummary(local);
    setDataSource('local');
    setLoading(false);

    // 2. Try to upgrade with server data in the background (3s timeout)
    try {
      const nodeServerUrl = process.env.REACT_APP_NODE_SERVER_URL || 'http://localhost:3000';
      const response = await axios.get(
        `${nodeServerUrl}/api/session/${sessionId}/summary`,
        { timeout: 6000 }
      );
      if (response.data.success && response.data.summary) {
        setSummary(response.data.summary);
        setDataSource('server');
      }
    } catch (error) {
      // Backend sleeping or no session — local summary is already shown, no problem
      console.warn('Server summary unavailable, using local data:', error.message);
    }
  }, [sessionId, interviewerName, sessionStartTime, localStressData, localAlerts]);

  useEffect(() => {
    fetchSummary();
  }, [fetchSummary]);

  const downloadPDF = async () => {
    setDownloading(true);
    try {
      const nodeServerUrl = process.env.REACT_APP_NODE_SERVER_URL || 'http://localhost:3000';
      const response = await axios.get(
        `${nodeServerUrl}/api/session/${sessionId}/export-pdf`,
        { responseType: 'blob', timeout: 30000 }
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
      // Fallback: generate simple text report from local data
      downloadLocalReport();
    } finally {
      setDownloading(false);
    }
  };

  const downloadLocalReport = () => {
    if (!summary) return;
    const stats = summary.stats;
    const fmt = (s) => `${Math.floor(s/60)}m ${Math.floor(s%60)}s`;
    const lines = [
      '===== INTERVIEW STRESS ANALYSIS REPORT =====',
      `Session ID : ${summary.session_id}`,
      `Interviewer: ${summary.interviewer}`,
      `Interviewee: ${summary.interviewee}`,
      `Start Time : ${new Date(summary.start_time).toLocaleString()}`,
      `End Time   : ${new Date(summary.end_time).toLocaleString()}`,
      `Duration   : ${fmt(stats.total_duration)}`,
      '',
      '--- KEY METRICS ---',
      `Avg Stress Score  : ${stats.avg_stress.toFixed(2)} (${stats.avg_stress < 1.3 ? 'Excellent' : stats.avg_stress < 1.6 ? 'Good' : 'High Stress'})`,
      `Avg Confidence    : ${(stats.avg_confidence * 100).toFixed(1)}%`,
      `Voice Confidence  : ${stats.avg_voice_confidence.toFixed(1)}%`,
      `Total Data Points : ${summary.total_data_points}`,
      `Total Alerts      : ${stats.total_alerts}`,
      '',
      '--- STRESS DISTRIBUTION ---',
      `Low Stress  : ${fmt(stats.low_stress_duration)}`,
      `High Stress : ${fmt(stats.high_stress_duration)}`,
      '',
      '--- ALERTS ---',
      ...(summary.alerts.length === 0
        ? ['No alerts triggered.']
        : summary.alerts.map(a =>
            `[${new Date(a.timestamp * 1000).toLocaleTimeString()}] ${a.message || a.type}`
          )
      ),
      '',
      '============================================'
    ];
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `interview_report_${sessionId}.txt`);
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  if (loading) {
    return (
      <div className="summary-modal">
        <div className="summary-content">
          <div className="loading-spinner"></div>
          <p>Building summary...</p>
        </div>
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="summary-modal">
        <div className="summary-content">
          <p>No data collected yet. Run the interview first.</p>
          <button onClick={onClose}>Close</button>
        </div>
      </div>
    );
  }

  const stats = summary.stats;

  return (
    <div className="summary-modal" onClick={onClose}>
      <div className="summary-content" onClick={(e) => e.stopPropagation()}>
        <div className="summary-header">
          <h2>📊 Interview Summary</h2>
          {dataSource === 'local' && (
            <span style={{fontSize:'11px', color:'#f59e0b', marginLeft:'8px'}}>⚡ Live data</span>
          )}
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
                    style={{ width: `${stats.total_duration > 0 ? (stats.low_stress_duration / stats.total_duration) * 100 : 0}%` }}
                  ></div>
                </div>
                <span className="stress-bar-value">{formatDuration(stats.low_stress_duration)}</span>
              </div>
              <div className="stress-bar-item">
                <span className="stress-bar-label">High Stress</span>
                <div className="stress-bar-container">
                  <div
                    className="stress-bar-fill high"
                    style={{ width: `${stats.total_duration > 0 ? (stats.high_stress_duration / stats.total_duration) * 100 : 0}%` }}
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
                      {new Date((alert.timestamp || Date.now()/1000) * 1000).toLocaleTimeString()}
                    </span>
                    <span className="alert-msg">{alert.message || alert.type}</span>
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
            {downloading ? 'Downloading...' : '📄 Download Report'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default InterviewSummary;
