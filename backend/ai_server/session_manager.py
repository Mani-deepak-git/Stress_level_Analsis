"""
Session Manager - Tracks interview sessions and collects analytics
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

class InterviewSession:
    def __init__(self, session_id: str, interviewer: str, interviewee: str):
        self.session_id = session_id
        self.interviewer = interviewer
        self.interviewee = interviewee
        self.start_time = time.time()
        self.end_time = None
        
        # Analytics data
        self.stress_data = []
        self.voice_confidence_data = []
        self.speech_metrics = []
        self.alerts = []
        
        # Summary stats
        self.stats = {
            'total_duration': 0,
            'avg_stress': 0,
            'avg_confidence': 0,
            'avg_voice_confidence': 0,
            'high_stress_duration': 0,
            'low_stress_duration': 0,
            'peak_stress_time': None,
            'total_alerts': 0
        }
    
    def add_stress_data(self, data: dict):
        """Add stress analysis data point"""
        self.stress_data.append({
            'timestamp': time.time(),
            'stress_level': data.get('stress_level'),
            'confidence_score': data.get('confidence_score'),
            'face_detected': data.get('face_detected', False)
        })
    
    def add_voice_confidence(self, data: dict):
        """Add voice confidence data point"""
        self.voice_confidence_data.append({
            'timestamp': time.time(),
            'confidence': data.get('confidence'),
            'stress_level': data.get('stress_level')
        })
    
    def add_speech_metric(self, data: dict):
        """Add speech analysis metric"""
        self.speech_metrics.append({
            'timestamp': time.time(),
            'speaking_pace': data.get('speaking_pace'),
            'pause_duration': data.get('pause_duration'),
            'speech_duration': data.get('speech_duration')
        })
    
    def add_alert(self, alert_type: str, message: str):
        """Add real-time alert"""
        self.alerts.append({
            'timestamp': time.time(),
            'type': alert_type,
            'message': message
        })
        self.stats['total_alerts'] += 1
    
    def end_session(self):
        """End session and calculate summary statistics"""
        self.end_time = time.time()
        self.stats['total_duration'] = self.end_time - self.start_time
        
        # Calculate averages
        if self.stress_data:
            stress_levels = [1 if d['stress_level'] in ['Low Stress', 'Medium Stress'] else 2 
                           for d in self.stress_data]
            self.stats['avg_stress'] = sum(stress_levels) / len(stress_levels)
            
            confidences = [d['confidence_score'] for d in self.stress_data]
            self.stats['avg_confidence'] = sum(confidences) / len(confidences)
            
            # High/Low stress duration
            high_stress_count = sum(1 for d in self.stress_data if d['stress_level'] == 'High Stress')
            self.stats['high_stress_duration'] = (high_stress_count / len(self.stress_data)) * self.stats['total_duration']
            self.stats['low_stress_duration'] = self.stats['total_duration'] - self.stats['high_stress_duration']
        
        if self.voice_confidence_data:
            voice_confs = [d['confidence'] for d in self.voice_confidence_data]
            self.stats['avg_voice_confidence'] = sum(voice_confs) / len(voice_confs)
    
    def get_summary(self) -> dict:
        """Get session summary"""
        return {
            'session_id': self.session_id,
            'interviewer': self.interviewer,
            'interviewee': self.interviewee,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            'stats': self.stats,
            'total_data_points': len(self.stress_data),
            'total_voice_points': len(self.voice_confidence_data),
            'total_speech_metrics': len(self.speech_metrics),
            'alerts': self.alerts
        }
    
    def get_full_data(self) -> dict:
        """Get complete session data for export"""
        return {
            **self.get_summary(),
            'stress_data': self.stress_data,
            'voice_confidence_data': self.voice_confidence_data,
            'speech_metrics': self.speech_metrics
        }


class SessionManager:
    def __init__(self):
        self.active_sessions: Dict[str, InterviewSession] = {}
        self.completed_sessions: Dict[str, InterviewSession] = {}
    
    def create_session(self, session_id: str, interviewer: str, interviewee: str) -> InterviewSession:
        """Create new interview session"""
        session = InterviewSession(session_id, interviewer, interviewee)
        self.active_sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[InterviewSession]:
        """Get active session"""
        return self.active_sessions.get(session_id)
    
    def end_session(self, session_id: str) -> Optional[dict]:
        """End session and move to completed"""
        session = self.active_sessions.pop(session_id, None)
        if session:
            session.end_session()
            self.completed_sessions[session_id] = session
            return session.get_summary()
        return None
    
    def get_completed_session(self, session_id: str) -> Optional[InterviewSession]:
        """Get completed session"""
        return self.completed_sessions.get(session_id)

# Global session manager
session_manager = SessionManager()
