"""
Speech Analysis - Analyzes speaking pace and pause patterns (numpy only, no librosa)
"""

import numpy as np
from collections import deque
import time

class SpeechAnalyzer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.speech_history = deque(maxlen=10)
        self.energy_threshold = 0.01

    def analyze_audio(self, audio_data: np.ndarray) -> dict:
        if audio_data is None or len(audio_data) == 0:
            return None

        y = np.array(audio_data, dtype=np.float32)
        if np.max(np.abs(y)) > 1.0:
            y = y / 32767.0

        total_duration = len(y) / self.sample_rate
        if total_duration < 0.1:
            return None

        # Frame-based RMS energy (25ms frames, 10ms hop)
        frame_len = int(0.025 * self.sample_rate)
        hop_len = int(0.010 * self.sample_rate)
        frames = []
        for i in range(0, len(y) - frame_len, hop_len):
            frame = y[i:i+frame_len]
            frames.append(np.sqrt(np.mean(frame ** 2)))

        if not frames:
            return None

        rms = np.array(frames)
        if np.max(rms) > 0:
            rms = rms / np.max(rms)

        speech_frames = rms > self.energy_threshold
        frame_duration = hop_len / self.sample_rate
        speech_duration = float(np.sum(speech_frames) * frame_duration)
        pause_duration = float(total_duration - speech_duration)
        pause_ratio = pause_duration / total_duration if total_duration > 0 else 0

        # Estimate speaking pace from speech segment transitions
        transitions = np.diff(speech_frames.astype(int))
        speech_segments = int(np.sum(transitions == 1))
        if speech_duration > 0.5:
            estimated_syllables = speech_duration / 0.2
            estimated_words = estimated_syllables / 1.5
            speaking_pace = float(np.clip((estimated_words / speech_duration) * 60, 0, 250))
        else:
            speaking_pace = 0.0

        result = {
            'speaking_pace': round(speaking_pace, 1),
            'pause_duration': round(pause_duration, 2),
            'speech_duration': round(speech_duration, 2),
            'pause_ratio': round(pause_ratio, 2),
            'total_duration': round(total_duration, 2),
            'timestamp': time.time()
        }
        self.speech_history.append(result)
        return result

    def reset(self):
        self.speech_history.clear()

# Global speech analyzer
speech_analyzer = SpeechAnalyzer()
