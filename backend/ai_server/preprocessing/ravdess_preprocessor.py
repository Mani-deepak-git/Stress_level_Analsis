import librosa
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class RAVDESSPreprocessor:
    def __init__(self, audio_dir, output_dir):
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.sample_rate = 16000
        self.max_length = 3.0  # 3 seconds max
        
        # Emotion to stress mapping (RAVDESS format: 01=neutral, 02=calm, etc.)
        self.emotion_to_stress = {
            1: 0,  # Neutral -> Low stress
            2: 0,  # Calm -> Low stress
            3: 0,  # Happy -> Low stress
            4: 2,  # Sad -> High stress
            5: 2,  # Angry -> High stress
            6: 2,  # Fearful -> High stress
            7: 1,  # Disgust -> Medium stress
            8: 1   # Surprised -> Medium stress
        }
    
    def extract_audio_features(self, audio_path):
        """Extract comprehensive audio features"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Trim silence
            y, _ = librosa.effects.trim(y, top_db=20)
            
            # Pad or truncate to fixed length
            max_samples = int(self.max_length * self.sample_rate)
            if len(y) > max_samples:
                y = y[:max_samples]
            else:
                y = np.pad(y, (0, max_samples - len(y)), mode='constant')
            
            # Extract features
            features = {}
            
            # 1. MFCC features (13 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            # 2. Pitch (F0)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
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
            
            # 3. Energy features
            rms = librosa.feature.rms(y=y)[0]
            features['energy_mean'] = np.mean(rms)
            features['energy_std'] = np.std(rms)
            
            # 4. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            # 5. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 6. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            features['chroma_std'] = np.std(chroma, axis=1)
            
            # Flatten all features
            feature_vector = []
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    feature_vector.extend(value)
                else:
                    feature_vector.append(value)
            
            return np.array(feature_vector)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def parse_filename(self, filename):
        """Parse RAVDESS filename to extract emotion"""
        # RAVDESS format: 03-01-06-01-02-01-12.wav
        # Position 3 (index 2) is emotion
        parts = filename.split('-')
        if len(parts) >= 3:
            emotion = int(parts[2])
            return emotion
        return None
    
    def process_dataset(self):
        """Process all audio files and extract features"""
        features_list = []
        emotions_list = []
        
        print("Processing RAVDESS audio files...")
        
        # Process all WAV files
        for root, dirs, files in os.walk(self.audio_dir):
            for file in files:
                if file.endswith('.wav'):
                    audio_path = os.path.join(root, file)
                    
                    # Extract emotion from filename
                    emotion = self.parse_filename(file)
                    if emotion is None:
                        continue
                    
                    # Extract features
                    features = self.extract_audio_features(audio_path)
                    if features is not None:
                        features_list.append(features)
                        emotions_list.append(emotion)
                        
                        if len(features_list) % 100 == 0:
                            print(f"Processed {len(features_list)} audio files...")
        
        print(f"Total processed files: {len(features_list)}")
        
        # Convert to numpy arrays
        X = np.array(features_list)
        emotions = np.array(emotions_list)
        
        # Map emotions to stress levels
        y = np.array([self.emotion_to_stress.get(emotion, 1) for emotion in emotions])
        
        return X, y
    
    def create_datasets(self):
        """Create and save preprocessed datasets"""
        # Process raw audio
        X, y = self.process_dataset()
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Feature dimension: {X_train.shape[1]}")
        print(f"Stress distribution - Train: {np.bincount(y_train)}")
        print(f"Stress distribution - Val: {np.bincount(y_val)}")
        
        # Save preprocessed data
        os.makedirs(self.output_dir, exist_ok=True)
        
        np.save(os.path.join(self.output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(self.output_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(self.output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(self.output_dir, 'y_val.npy'), y_val)
        
        # Save scaler for inference
        with open(os.path.join(self.output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"Preprocessed data saved to {self.output_dir}")
        
        return X_train, X_val, y_train, y_val

class RAVDESSDataset(Dataset):
    """PyTorch Dataset for RAVDESS"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def create_audio_data_loaders(data_dir, batch_size=32):
    """Create PyTorch data loaders for audio data"""
    
    # Load preprocessed data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    
    # Create datasets
    train_dataset = RAVDESSDataset(X_train, y_train)
    val_dataset = RAVDESSDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Example usage
    audio_dir = "../../datasets/ravdess/audio_speech_actors_01-24"
    output_dir = "../../datasets/ravdess/preprocessed"
    
    preprocessor = RAVDESSPreprocessor(audio_dir, output_dir)
    X_train, X_val, y_train, y_val = preprocessor.create_datasets()
    
    print("RAVDESS preprocessing complete!")