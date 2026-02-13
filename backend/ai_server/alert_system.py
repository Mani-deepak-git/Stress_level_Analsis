"""
Real-Time Alert System - Monitors stress and triggers alerts
"""

from collections import deque
import time

class AlertSystem:
    def __init__(self):
        # Alert thresholds
        self.high_stress_threshold = 1.8  # Stress level value
        self.low_confidence_threshold = 0.3
        self.low_voice_confidence_threshold = 30
        
        # Duration thresholds (seconds)
        self.high_stress_duration_threshold = 30  # Alert after 30s of high stress
        self.low_confidence_duration_threshold = 45
        
        # History tracking
        self.stress_history = deque(maxlen=15)  # Last 15 data points (~30s at 2s intervals)
        self.confidence_history = deque(maxlen=15)
        self.voice_confidence_history = deque(maxlen=15)
        
        # Alert cooldown (prevent spam)
        self.last_alert_time = {}
        self.alert_cooldown = 60  # 60 seconds between same alert type
        
        # Alert counters
        self.alert_counts = {
            'high_stress': 0,
            'low_confidence': 0,
            'low_voice_confidence': 0,
            'no_face': 0,
            'prolonged_stress': 0
        }
    
    def check_stress_alert(self, stress_level: str, confidence_score: float, face_detected: bool) -> list:
        """
        Check stress data and return alerts if needed
        Returns: list of alert dicts
        """
        alerts = []
        current_time = time.time()
        
        # Convert stress level to numeric
        stress_value = 2 if stress_level == 'High Stress' else 1
        
        # Add to history
        self.stress_history.append({
            'value': stress_value,
            'time': current_time
        })
        self.confidence_history.append({
            'value': confidence_score,
            'time': current_time
        })
        
        # Check for prolonged high stress
        if len(self.stress_history) >= 10:
            recent_stress = [h['value'] for h in list(self.stress_history)[-10:]]
            high_stress_count = sum(1 for s in recent_stress if s >= 1.8)
            
            if high_stress_count >= 8:  # 80% of last 10 readings
                alert = self._create_alert(
                    'prolonged_stress',
                    'High Stress Detected',
                    'Candidate has been under high stress for 30+ seconds. Consider taking a break.',
                    'high',
                    current_time
                )
                if alert:
                    alerts.append(alert)
        
        # Check for immediate high stress
        if stress_value >= self.high_stress_threshold:
            alert = self._create_alert(
                'high_stress',
                'High Stress Alert',
                'Candidate is experiencing high stress levels.',
                'medium',
                current_time
            )
            if alert:
                alerts.append(alert)
        
        # Check for low confidence
        if confidence_score < self.low_confidence_threshold:
            alert = self._create_alert(
                'low_confidence',
                'Low Confidence Detected',
                'Candidate showing low confidence. Try encouraging questions.',
                'medium',
                current_time
            )
            if alert:
                alerts.append(alert)
        
        # Check for no face detection
        if not face_detected:
            alert = self._create_alert(
                'no_face',
                'Face Not Detected',
                'Camera may be blocked or candidate moved out of frame.',
                'low',
                current_time
            )
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def check_voice_confidence_alert(self, voice_confidence: float) -> list:
        """Check voice confidence and return alerts"""
        alerts = []
        current_time = time.time()
        
        self.voice_confidence_history.append({
            'value': voice_confidence,
            'time': current_time
        })
        
        # Check for consistently low voice confidence
        if len(self.voice_confidence_history) >= 10:
            recent_voice = [h['value'] for h in list(self.voice_confidence_history)[-10:]]
            avg_voice = sum(recent_voice) / len(recent_voice)
            
            if avg_voice < self.low_voice_confidence_threshold:
                alert = self._create_alert(
                    'low_voice_confidence',
                    'Low Voice Confidence',
                    f'Voice confidence averaging {avg_voice:.1f}%. Candidate may be nervous or uncertain.',
                    'medium',
                    current_time
                )
                if alert:
                    alerts.append(alert)
        
        return alerts
    
    def check_speech_alert(self, speech_metrics: dict) -> list:
        """Check speech patterns and return alerts"""
        alerts = []
        current_time = time.time()
        
        # Check for very slow speaking (may indicate hesitation)
        if speech_metrics.get('speaking_pace', 0) < 60:
            alert = self._create_alert(
                'slow_speech',
                'Slow Speaking Pace',
                'Candidate speaking very slowly. May indicate uncertainty or careful thinking.',
                'low',
                current_time
            )
            if alert:
                alerts.append(alert)
        
        # Check for excessive pauses
        if speech_metrics.get('pause_ratio', 0) > 0.6:
            alert = self._create_alert(
                'excessive_pauses',
                'Frequent Pauses',
                'Candidate taking frequent pauses. May need more time to think.',
                'low',
                current_time
            )
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def _create_alert(self, alert_type: str, title: str, message: str, severity: str, current_time: float) -> dict:
        """
        Create alert if cooldown period has passed
        Returns: alert dict or None
        """
        last_time = self.last_alert_time.get(alert_type, 0)
        
        if current_time - last_time >= self.alert_cooldown:
            self.last_alert_time[alert_type] = current_time
            self.alert_counts[alert_type] = self.alert_counts.get(alert_type, 0) + 1
            
            return {
                'type': alert_type,
                'title': title,
                'message': message,
                'severity': severity,  # 'low', 'medium', 'high'
                'timestamp': current_time
            }
        
        return None
    
    def get_alert_summary(self) -> dict:
        """Get summary of all alerts"""
        return {
            'total_alerts': sum(self.alert_counts.values()),
            'alert_breakdown': self.alert_counts.copy()
        }
    
    def reset(self):
        """Reset alert system"""
        self.stress_history.clear()
        self.confidence_history.clear()
        self.voice_confidence_history.clear()
        self.last_alert_time.clear()
        self.alert_counts = {k: 0 for k in self.alert_counts}

# Global alert system
alert_system = AlertSystem()
