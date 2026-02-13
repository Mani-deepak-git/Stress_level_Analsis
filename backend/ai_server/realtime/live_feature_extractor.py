"""
Live Feature Extractor
Extracts audio features from live audio matching RAVDESS training preprocessing
Why: Must use IDENTICAL features as training for model compatibility
"""

import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

class LiveFeatureExtractor:
    """
    Extracts audio features matching RAVDESS preprocessing
    Why: Model expects exact same feature format as training
    """
    
    def __init__(self, sample_rate=22050, scaler_path=None):
        """
        Args:
            sample_rate: Must match training (22050 Hz)
            scaler_path: Path to saved StandardScaler from training
        """
        self.sample_rate = sample_rate
        self.scaler = None
        
        # Load scaler if available (critical for matching training normalization)
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Loaded scaler from {scaler_path}")
        else:
            print("Warning: No scaler loaded. Features may not match training distribution")
    
    def extract_features(self, audio):
        """
        Extract features matching RAVDESS preprocessing
        Why: Must match training features exactly
        
        Args:
            audio: numpy array of audio samples
            
        Returns:
            Feature vector matching training format
        """
        try:
            # Ensure audio is float32
            y = audio.astype(np.float32)
            
            # Trim silence (matching training)
            y, _ = librosa.effects.trim(y, top_db=20)
            
            if len(y) == 0:
                return None
            
            features = {}
            
            # 1. MFCC features (13 coefficients) - CRITICAL
            mfccs = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            # 2. Pitch (F0) - CRITICAL
            pitches, magnitudes = librosa.piptrack(y=y, sr=self.sample_rate)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
            
            # 3. Energy features - CRITICAL
            rms = librosa.feature.rms(y=y)[0]
            features['energy_mean'] = np.mean(rms)
            features['energy_std'] = np.std(rms)
            
            # 4. Spectral features - CRITICAL
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            # 5. Zero crossing rate - CRITICAL
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 6. Chroma features - CRITICAL
            chroma = librosa.feature.chroma_stft(y=y, sr=self.sample_rate)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            features['chroma_std'] = np.std(chroma, axis=1)
            
            # Flatten all features (matching training format)
            feature_vector = []
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    feature_vector.extend(value)
                else:
                    feature_vector.append(value)
            
            feature_vector = np.array(feature_vector)
            
            # Apply scaler (CRITICAL - must match training normalization)
            if self.scaler:
                feature_vector = self.scaler.transform([feature_vector])[0]
            
            return feature_vector
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def get_feature_dimension(self):
        """
        Get expected feature dimension
        Why: Model input size must match
        """
        # MFCC: 13*2=26, Pitch: 2, Energy: 2, Spectral: 4, ZCR: 2, Chroma: 12*2=24
        # Total: 26+2+2+4+2+24 = 60 features
        return 60


if __name__ == "__main__":
    # Test feature extraction
    print("Testing feature extraction...")
    
    # Generate test audio
    duration = 2.0
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    extractor = LiveFeatureExtractor(sample_rate=sample_rate)
    features = extractor.extract_features(test_audio)
    
    if features is not None:
        print(f"Feature shape: {features.shape}")
        print(f"Expected dimension: {extractor.get_feature_dimension()}")
        print(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")
    
    print("Test complete")
