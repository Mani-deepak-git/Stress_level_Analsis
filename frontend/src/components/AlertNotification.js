import React, { useState, useEffect } from 'react';
import './AlertNotification.css';

const AlertNotification = ({ alert, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose();
    }, 8000); // Auto-dismiss after 8 seconds

    return () => clearTimeout(timer);
  }, [onClose]);

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return '#ef4444';
      case 'medium': return '#f97316';
      case 'low': return '#3b82f6';
      default: return '#64748b';
    }
  };

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'high': return '🚨';
      case 'medium': return '⚠️';
      case 'low': return 'ℹ️';
      default: return '📢';
    }
  };

  return (
    <div 
      className="alert-notification"
      style={{ borderLeft: `4px solid ${getSeverityColor(alert.severity)}` }}
    >
      <div className="alert-icon">{getSeverityIcon(alert.severity)}</div>
      <div className="alert-content">
        <div className="alert-title">{alert.title}</div>
        <div className="alert-message">{alert.message}</div>
      </div>
      <button className="alert-close" onClick={onClose}>×</button>
    </div>
  );
};

const AlertContainer = ({ alerts, onDismiss }) => {
  return (
    <div className="alert-container">
      {alerts.map((alert, index) => (
        <AlertNotification
          key={`${alert.timestamp}-${index}`}
          alert={alert}
          onClose={() => onDismiss(index)}
        />
      ))}
    </div>
  );
};

export default AlertContainer;
