"""
Speech Analysis - Analyzes speaking pace and pause patterns
"""

import numpy as np
import librosa
from collections import deque
import time

class SpeechAnalyzer:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.speech_history = deque(maxlen=10)
        
        # Thresholds
        self.energy_threshold = 0.02  # Minimum energy to consider as speech
        self.min_speech_duration = 0.3  # Minimum duration for speech segment (seconds)
        self.min_pause_duration = 0.2  # Minimum duration for pause (seconds)
        
    def analyze_audio(self, audio_data: np.ndarray) -> dict:
        """
        Analyze audio for speech patterns
        Returns: dict with speaking_pace, pause_duration, speech_duration
        """
        if audio_data is None or len(audio_data) == 0:
            return None
        
        # Calculate energy (RMS)
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        rms = librosa.feature.rms(
            y=audio_data,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Normalize RMS
        if np.max(rms) > 0:
            rms = rms / np.max(rms)
        
        # Detect speech segments
        speech_frames = rms > self.energy_threshold
        
        # Calculate speech and pause durations
        frame_duration = hop_length / self.sample_rate
        total_duration = len(audio_data) / self.sample_rate
        
        speech_duration = np.sum(speech_frames) * frame_duration
        pause_duration = total_duration - speech_duration
        
        # Calculate speaking pace (words per minute estimate)
        # Rough estimate: average syllable rate is 4-5 per second for normal speech
        # We estimate based on speech segments
        speaking_pace = self._estimate_speaking_pace(speech_frames, frame_duration)
        
        # Calculate pause ratio
        pause_ratio = pause_duration / total_duration if total_duration > 0 else 0
        
        result = {
            'speaking_pace': round(speaking_pace, 1),  # words per minute
            'pause_duration': round(pause_duration, 2),  # seconds
            'speech_duration': round(speech_duration, 2),  # seconds
            'pause_ratio': round(pause_ratio, 2),  # percentage
            'total_duration': round(total_duration, 2),
            'timestamp': time.time()
        }
        
        self.speech_history.append(result)
        return result
    
    def _estimate_speaking_pace(self, speech_frames: np.ndarray, frame_duration: float) -> float:
        """
        Estimate speaking pace in words per minute
        Based on speech segment transitions
        """
        # Count speech segments (transitions from silence to speech)
        transitions = np.diff(speech_frames.astype(int))
        speech_segments = np.sum(transitions == 1)
        
        # Estimate syllables (rough approximation)
        # Average: 1 syllable per 0.2 seconds of speech
        total_speech_time = np.sum(speech_frames) * frame_duration
        
        if total_speech_time < 0.5:  # Too short to estimate
            return 0
        
        # Rough conversion: syllables to words (avg 1.5 syllables per word)
        # Normal speaking pace: 120-150 words per minute
        estimated_syllables = total_speech_time / 0.2
        estimated_words = estimated_syllables / 1.5
        
        # Convert to words per minute
        words_per_minute = (estimated_words / total_speech_time) * 60
        
        # Clamp to reasonable range
        return np.clip(words_per_minute, 0, 250)
    
    def get_average_metrics(self) -> dict:
        """Get average metrics from history"""
        if not self.speech_history:
            return None
        
        avg_pace = np.mean([h['speaking_pace'] for h in self.speech_history])
        avg_pause = np.mean([h['pause_duration'] for h in self.speech_history])
        avg_speech = np.mean([h['speech_duration'] for h in self.speech_history])
        avg_pause_ratio = np.mean([h['pause_ratio'] for h in self.speech_history])
        
        return {
            'avg_speaking_pace': round(avg_pace, 1),
            'avg_pause_duration': round(avg_pause, 2),
            'avg_speech_duration': round(avg_speech, 2),
            'avg_pause_ratio': round(avg_pause_ratio, 2),
            'interpretation': self._interpret_metrics(avg_pace, avg_pause_ratio)
        }
    
    def _interpret_metrics(self, pace: float, pause_ratio: float) -> str:
        """Interpret speech metrics"""
        if pace < 80:
            pace_desc = "slow"
        elif pace < 160:
            pace_desc = "normal"
        else:
            pace_desc = "fast"
        
        if pause_ratio < 0.2:
            pause_desc = "minimal pauses"
        elif pause_ratio < 0.4:
            pause_desc = "normal pauses"
        else:
            pause_desc = "frequent pauses"
        
        return f"Speaking pace is {pace_desc} with {pause_desc}"
    
    def reset(self):
        """Reset analyzer history"""
        self.speech_history.clear()

# Global speech analyzer
speech_analyzer = SpeechAnalyzer()
