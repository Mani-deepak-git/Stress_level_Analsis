import torch
import cv2
import numpy as np
import librosa
import pickle
import os
from collections import deque
import mediapipe as mp
from models.face_model import FaceStressCNN
from models.voice_model import VoiceStressLSTM
from models.fusion_model import MultimodalFusionModel

class RealTimeStressAnalyzer:
    """Real-time stress and confidence analyzer"""
    
    def __init__(self, models_dir="../../models/trained", device='cpu'):
        self.device = device
        self.models_dir = models_dir
        
        # Load models
        self.face_model = self._load_face_model()
        self.voice_model = self._load_voice_model()
        self.use_fusion = self._try_load_fusion_model()
        self.audio_scaler = self._load_audio_scaler()
        
        # MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.audio_buffer = deque(maxlen=int(5 * 44100))  # 5 seconds at browser sample rate
        self.min_audio_samples = int(2 * 44100)  # 2 seconds at browser sample rate
        
        # Temporal smoothing
        self.stress_history = deque(maxlen=5)  # Last 5 predictions
        self.confidence_history = deque(maxlen=5)
        
        # Stress level mapping
        self.stress_labels = ['Low Stress', 'Medium Stress', 'High Stress']
        
        print(f"Real-time stress analyzer initialized! Fusion model: {'Enabled' if self.use_fusion else 'Disabled'}")
    
    def _load_face_model(self):
        """Load pre-trained face model"""
        model_path = os.path.join(self.models_dir, "face_stress_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Face model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model = FaceStressCNN(num_classes=3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        
        print("Face model loaded successfully")
        return model
    
    def _load_voice_model(self):
        """Load pre-trained voice model"""
        model_path = os.path.join(self.models_dir, "voice_stress_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Voice model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model = VoiceStressLSTM(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            num_classes=3
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        
        print("Voice model loaded successfully")
        return model
    
    def _try_load_fusion_model(self):
        """Try to load fusion model, return False if not available"""
        model_path = os.path.join(self.models_dir, "fusion_model.pth")
        if not os.path.exists(model_path):
            print("Fusion model not found. Using individual models.")
            self.fusion_model = None
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.fusion_model = MultimodalFusionModel(
                face_feature_dim=checkpoint['face_feature_dim'],
                voice_feature_dim=checkpoint['voice_feature_dim'],
                num_classes=3
            )
            self.fusion_model.load_state_dict(checkpoint['model_state_dict'])
            self.fusion_model.eval()
            self.fusion_model.to(self.device)
            
            print("Fusion model loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load fusion model: {e}. Using individual models.")
            self.fusion_model = None
            return False
    
    def _load_audio_scaler(self):
        """Load audio feature scaler"""
        scaler_path = "../../datasets/ravdess/preprocessed/scaler.pkl"
        if not os.path.exists(scaler_path):
            print("WARNING: Audio scaler not found. Using default normalization.")
            return None
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print("Audio scaler loaded successfully")
        return scaler
    
    def preprocess_face_frame(self, frame):
        """Preprocess face frame for model input"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                # Get first detected face
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Extract face region
                face_region = frame[y:y+height, x:x+width]
                
                if face_region.size > 0:
                    # Convert to grayscale and resize
                    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                    resized_face = cv2.resize(gray_face, (48, 48))
                    
                    # Normalize
                    normalized_face = resized_face.astype(np.float32) / 255.0
                    
                    # Add batch and channel dimensions
                    face_tensor = torch.FloatTensor(normalized_face).unsqueeze(0).unsqueeze(0)
                    
                    return face_tensor.to(self.device)
            
            return None
            
        except Exception as e:
            print(f"Error preprocessing face frame: {e}")
            return None
    
    def extract_audio_features(self, audio_data):
        """Extract audio features from audio data"""
        try:
            # Convert to numpy array and normalize
            y = np.array(audio_data, dtype=np.float32)
            
            # Normalize if needed (audio should be in [-1, 1] range)
            if np.max(np.abs(y)) > 1.0:
                y = y / 32767.0  # Convert from 16-bit to float
            
            # Resample to 16kHz if needed (most browsers use 44.1kHz)
            original_sr = 44100  # Assume browser sample rate
            if len(y) > 0:
                y_resampled = librosa.resample(y, orig_sr=original_sr, target_sr=self.sample_rate)
                y = y_resampled
            
            # Take at least 1 second of audio
            if len(y) < self.sample_rate:
                return None, None
            
            # Take first second for analysis
            y = y[:self.sample_rate]
            
            # Extract features (same as training)
            features = {}
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            # Pitch
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
            
            # Energy features
            rms = librosa.feature.rms(y=y)[0]
            features['energy_mean'] = np.mean(rms)
            features['energy_std'] = np.std(rms)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=self.sample_rate)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            features['chroma_std'] = np.std(chroma, axis=1)
            
            # Flatten features
            feature_vector = []
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    feature_vector.extend(value)
                else:
                    feature_vector.append(value)
            
            feature_vector = np.array(feature_vector)
            
            # Apply scaler if available
            if self.audio_scaler:
                feature_vector = self.audio_scaler.transform([feature_vector])[0]
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(feature_vector).unsqueeze(0)
            
            return audio_tensor.to(self.device), features
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return None, None
    
    def calculate_voice_confidence(self, audio_features, y):
        """Calculate confidence based on voice characteristics"""
        try:
            # Debug: Print energy values
            energy_mean = audio_features.get('energy_mean', 0)
            print(f"DEBUG: Energy mean = {energy_mean}")
            
            # Much lower thresholds for browser audio
            if energy_mean > 0.001:  # Very low threshold for speaking
                confidence_score = 0.8
                print(f"DEBUG: Speaking detected, confidence = {confidence_score}")
            elif energy_mean > 0.0005:  # Even lower for quiet
                confidence_score = 0.6
                print(f"DEBUG: Quiet speaking, confidence = {confidence_score}")
            else:
                confidence_score = 0.3
                print(f"DEBUG: Silent, confidence = {confidence_score}")
            
            return confidence_score
            
        except Exception as e:
            print(f"Error calculating voice confidence: {e}")
            return 0.5
    
    def analyze_frame(self, video_frame, audio_chunk=None):
        """Analyze single frame with optional audio"""
        print(f"\n=== ANALYZE FRAME CALLED ===")
        print(f"Audio chunk received: {audio_chunk is not None}")
        if audio_chunk is not None:
            print(f"Audio chunk length: {len(audio_chunk)}")
            print(f"Audio chunk sample values (first 10): {audio_chunk[:10] if len(audio_chunk) >= 10 else audio_chunk}")
        
        results = {
            'stress_level': 'Unknown',
            'stress_probability': [0.33, 0.33, 0.34],
            'confidence_score': 0.5,
            'face_detected': False,
            'audio_processed': False
        }
        
        try:
            face_stress_probs = None
            voice_stress_probs = None
            
            # Process video frame
            face_tensor = self.preprocess_face_frame(video_frame)
            
            if face_tensor is not None:
                results['face_detected'] = True
                
                # Get face prediction
                with torch.no_grad():
                    face_logits = self.face_model(face_tensor)
                    face_stress_probs = torch.softmax(face_logits, dim=1).cpu().numpy()[0]
            
            # Process audio - check buffer even without new chunk
            if audio_chunk is not None:
                print(f"DEBUG: Received audio chunk with {len(audio_chunk)} samples")
                self.audio_buffer.extend(audio_chunk)
            
            print(f"DEBUG: Audio buffer size: {len(self.audio_buffer)}, min required: {self.min_audio_samples}")
            
            # Try to process audio if we have enough in buffer
            if len(self.audio_buffer) >= self.min_audio_samples:
                print("DEBUG: Processing audio buffer...")
                audio_result = self.extract_audio_features(list(self.audio_buffer))
                
                if audio_result[0] is not None:
                    audio_tensor, audio_features = audio_result
                    results['audio_processed'] = True
                    print("DEBUG: Audio features extracted successfully")
                    
                    # Get voice prediction
                    with torch.no_grad():
                        voice_logits = self.voice_model(audio_tensor)
                        voice_stress_probs = torch.softmax(voice_logits, dim=1).cpu().numpy()[0]
                    
                    # Calculate voice-based confidence
                    y = np.array(list(self.audio_buffer), dtype=np.float32)
                    if np.max(np.abs(y)) > 1.0:
                        y = y / 32767.0
                    y_resampled = librosa.resample(y, orig_sr=44100, target_sr=self.sample_rate)
                    voice_confidence = self.calculate_voice_confidence(audio_features, y_resampled[:self.sample_rate])
                else:
                    print("DEBUG: Audio feature extraction failed")
                    voice_stress_probs = None
                    voice_confidence = None
            else:
                print("DEBUG: Not enough audio data in buffer")
                voice_stress_probs = None
                voice_confidence = None
            
            # Use face for stress level, voice for confidence
            if face_stress_probs is not None:
                # Stress level comes from face detection
                stress_probs = face_stress_probs.copy()
                
                # Confidence based on audio buffer activity (not just incoming chunk)
                if len(self.audio_buffer) > 1000:  # At least some audio data
                    # Check recent audio activity from buffer
                    recent_audio = list(self.audio_buffer)[-4410:]  # Last 0.1 seconds
                    audio_array = np.array(recent_audio, dtype=np.float32)
                    audio_energy = np.mean(np.abs(audio_array))
                    
                    if audio_energy > 0.01:  # Speaking
                        confidence_score = 0.9
                    elif audio_energy > 0.005:  # Quiet speaking
                        confidence_score = 0.7
                    elif audio_energy > 0.001:  # Very quiet
                        confidence_score = 0.5
                    else:  # Silent
                        confidence_score = 0.3
                    
                    print(f"DEBUG: Audio energy = {audio_energy:.6f}, confidence = {confidence_score}")
                else:
                    # No audio data in buffer
                    confidence_score = 0.3
                    print(f"DEBUG: No audio in buffer, confidence = {confidence_score}")
            else:
                # No face detected, fallback to default
                stress_probs = np.array([0.33, 0.33, 0.34])
                confidence_score = 0.3
            
            # Get stress level
            stress_level_idx = np.argmax(stress_probs)
            
            # Apply temporal smoothing ONLY to stress, not confidence
            self.stress_history.append(stress_level_idx)
            
            # Smooth stress predictions only
            if len(self.stress_history) > 1:
                smoothed_stress_idx = int(np.round(np.mean(self.stress_history)))
            else:
                smoothed_stress_idx = stress_level_idx
            
            # Update results - use raw confidence, not smoothed
            results['stress_level'] = self.stress_labels[smoothed_stress_idx]
            results['stress_probability'] = stress_probs.tolist()
            results['confidence_score'] = float(confidence_score)
            
        except Exception as e:
            print(f"Error in frame analysis: {e}")
        
        return results
    
    def reset_history(self):
        """Reset temporal smoothing history"""
        self.stress_history.clear()
        self.confidence_history.clear()
        self.audio_buffer.clear()
        print("Analysis history reset")

# Utility functions for integration
def create_analyzer():
    """Factory function to create analyzer instance"""
    try:
        analyzer = RealTimeStressAnalyzer()
        return analyzer
    except Exception as e:
        print(f"Error creating analyzer: {e}")
        return None

def analyze_video_frame(analyzer, frame):
    """Analyze single video frame"""
    if analyzer is None:
        return None
    return analyzer.analyze_frame(frame)

def analyze_multimodal(analyzer, frame, audio_chunk):
    """Analyze video frame with audio"""
    if analyzer is None:
        return None
    return analyzer.analyze_frame(frame, audio_chunk)

if __name__ == "__main__":
    # Test the analyzer
    print("Testing Real-time Stress Analyzer...")
    
    try:
        analyzer = create_analyzer()
        if analyzer:
            print("Analyzer created successfully!")
            
            # Test with dummy data
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy_audio = np.random.randn(16000).tolist()  # 1 second of audio
            
            results = analyze_multimodal(analyzer, dummy_frame, dummy_audio)
            print("Test results:", results)
        else:
            print("Failed to create analyzer")
            
    except Exception as e:
        print(f"Test failed: {e}")