"""
Live Audio Capture Module
Captures real-time audio from microphone using sounddevice
Uses sliding window approach for continuous streaming
"""

import sounddevice as sd
import numpy as np
from collections import deque
import threading
import time

class LiveAudioCapture:
    """
    Captures live audio from microphone with sliding window
    Why: Provides continuous audio stream for real-time analysis
    """
    
    def __init__(self, sample_rate=22050, window_duration=2.0, overlap=0.5):
        """
        Args:
            sample_rate: Must match training (22050 Hz for RAVDESS)
            window_duration: Audio window size in seconds (1-2s recommended)
            overlap: Overlap ratio for sliding window (0.5 = 50% overlap)
        """
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.overlap = overlap
        
        # Calculate window size in samples
        self.window_size = int(sample_rate * window_duration)
        self.hop_size = int(self.window_size * (1 - overlap))
        
        # Audio buffer (thread-safe)
        self.buffer = deque(maxlen=self.window_size * 2)
        self.lock = threading.Lock()
        
        # Control flags
        self.is_capturing = False
        self.stream = None
        
        print(f"Audio Capture initialized: {sample_rate}Hz, {window_duration}s window, {overlap*100}% overlap")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback for sounddevice stream
        Why: Continuously receives audio chunks from microphone
        """
        if status:
            print(f"Audio status: {status}")
        
        # Add audio to buffer (thread-safe)
        with self.lock:
            self.buffer.extend(indata[:, 0])  # Mono channel
    
    def start_capture(self):
        """
        Start capturing audio from microphone
        Why: Initiates continuous audio stream
        """
        if self.is_capturing:
            print("Already capturing")
            return
        
        self.is_capturing = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,  # Mono
            callback=self._audio_callback,
            blocksize=self.hop_size
        )
        self.stream.start()
        print("Audio capture started")
    
    def stop_capture(self):
        """Stop capturing audio"""
        if not self.is_capturing:
            return
        
        self.is_capturing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("Audio capture stopped")
    
    def get_audio_window(self):
        """
        Get current audio window for processing
        Why: Provides audio chunk for feature extraction
        
        Returns:
            numpy array of shape (window_size,) or None if insufficient data
        """
        with self.lock:
            if len(self.buffer) < self.window_size:
                return None
            
            # Get last window_size samples
            audio_window = np.array(list(self.buffer)[-self.window_size:])
            return audio_window
    
    def clear_buffer(self):
        """Clear audio buffer"""
        with self.lock:
            self.buffer.clear()


if __name__ == "__main__":
    # Test audio capture
    print("Testing live audio capture...")
    
    capture = LiveAudioCapture(sample_rate=22050, window_duration=2.0)
    capture.start_capture()
    
    print("Recording for 5 seconds...")
    time.sleep(5)
    
    # Get audio window
    audio = capture.get_audio_window()
    if audio is not None:
        print(f"Captured audio shape: {audio.shape}")
        print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
    
    capture.stop_capture()
    print("Test complete")
