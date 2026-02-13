"""
Complete Model Training Pipeline
Trains Face CNN -> Voice LSTM -> Fusion Model sequentially
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.fer2013_preprocessor import FER2013Preprocessor
from preprocessing.fer2013_folder_preprocessor import FER2013FolderPreprocessor
from preprocessing.ravdess_preprocessor import RAVDESSPreprocessor
from models.face_model import train_face_model
from models.voice_model import train_voice_model
from models.fusion_model import train_fusion_model

def setup_directories():
    """Create necessary directories"""
    directories = [
        "../../datasets/fer2013/preprocessed",
        "../../datasets/ravdess/preprocessed", 
        "../../models/trained"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def preprocess_datasets(fer2013_csv_path, ravdess_audio_dir, force_preprocessing=False):
    """Preprocess both datasets (only if not already preprocessed)"""
    print("="*60)
    print("STEP 1: PREPROCESSING DATASETS")
    print("="*60)
    
    # Check if FER-2013 preprocessing already exists
    fer2013_preprocessed_dir = "../../datasets/fer2013/preprocessed"
    fer2013_files = ['X_train.npy', 'X_val.npy', 'y_train.npy', 'y_val.npy']
    fer2013_exists = all(os.path.exists(os.path.join(fer2013_preprocessed_dir, f)) for f in fer2013_files)
    
    if fer2013_exists and not force_preprocessing:
        print("FER-2013 preprocessed data found. Skipping preprocessing.")
    else:
        if force_preprocessing and fer2013_exists:
            print("Force preprocessing enabled. Reprocessing FER-2013...")
        # Preprocess FER-2013
        fer2013_folder_path = "../../datasets/fer2013"
        
        if os.path.exists(fer2013_csv_path):
            print("Preprocessing FER-2013 dataset from CSV...")
            fer_preprocessor = FER2013Preprocessor(
                csv_path=fer2013_csv_path,
                output_dir=fer2013_preprocessed_dir
            )
            fer_preprocessor.create_datasets()
            print("FER-2013 preprocessing complete!")
        elif os.path.exists(os.path.join(fer2013_folder_path, "train")) or any(os.path.exists(os.path.join(fer2013_folder_path, emotion)) for emotion in ["angry", "happy", "sad", "neutral", "fear", "surprise", "disgust"]):
            print("Preprocessing FER-2013 dataset from folders...")
            fer_preprocessor = FER2013FolderPreprocessor(
                dataset_dir=fer2013_folder_path,
                output_dir=fer2013_preprocessed_dir
            )
            fer_preprocessor.create_datasets()
            print("FER-2013 preprocessing complete!")
        else:
            print(f"WARNING: FER-2013 data not found at {fer2013_csv_path} or {fer2013_folder_path}")
            print("Please download FER-2013 dataset from Kaggle and place it in datasets/fer2013/")
    
    # Check if RAVDESS preprocessing already exists
    ravdess_preprocessed_dir = "../../datasets/ravdess/preprocessed"
    ravdess_files = ['X_train.npy', 'X_val.npy', 'y_train.npy', 'y_val.npy']
    ravdess_exists = all(os.path.exists(os.path.join(ravdess_preprocessed_dir, f)) for f in ravdess_files)
    
    if ravdess_exists and not force_preprocessing:
        print("RAVDESS preprocessed data found. Skipping preprocessing.")
    else:
        if force_preprocessing and ravdess_exists:
            print("Force preprocessing enabled. Reprocessing RAVDESS...")
        # Preprocess RAVDESS
        if os.path.exists(ravdess_audio_dir):
            print("Preprocessing RAVDESS dataset...")
            ravdess_preprocessor = RAVDESSPreprocessor(
                audio_dir=ravdess_audio_dir,
                output_dir=ravdess_preprocessed_dir
            )
            ravdess_preprocessor.create_datasets()
            print("RAVDESS preprocessing complete!")
        else:
            print(f"WARNING: RAVDESS audio directory not found at {ravdess_audio_dir}")
            print("Please download RAVDESS dataset from Kaggle and extract to datasets/ravdess/")

def train_models(force_training=False):
    """Train all models sequentially (skip if already trained)"""
    print("="*60)
    print("STEP 2: TRAINING MODELS")
    print("="*60)
    
    # Check if preprocessed data exists
    face_data_dir = "../../datasets/fer2013/preprocessed"
    voice_data_dir = "../../datasets/ravdess/preprocessed"
    
    face_data_exists = all([
        os.path.exists(os.path.join(face_data_dir, f))
        for f in ['X_train.npy', 'X_val.npy', 'y_train.npy', 'y_val.npy']
    ])
    
    voice_data_exists = all([
        os.path.exists(os.path.join(voice_data_dir, f))
        for f in ['X_train.npy', 'X_val.npy', 'y_train.npy', 'y_val.npy']
    ])
    
    # Model paths
    face_model_path = "../../models/trained/face_stress_model.pth"
    voice_model_path = "../../models/trained/voice_stress_model.pth"
    fusion_model_path = "../../models/trained/fusion_model.pth"
    
    # Train Face Model
    if os.path.exists(face_model_path) and not force_training:
        print("\n" + "-"*40)
        print("Face model already exists. Skipping training.")
        print("-"*40)
    elif face_data_exists:
        if force_training and os.path.exists(face_model_path):
            print("\n" + "-"*40)
            print("Force training enabled. Retraining Face model...")
            print("-"*40)
        else:
            print("\n" + "-"*40)
            print("Training Face Stress CNN...")
            print("-"*40)
        
        face_model, face_acc = train_face_model(
            data_dir=face_data_dir,
            model_save_path=face_model_path,
            epochs=30
        )
        print(f"Face model training complete! Accuracy: {face_acc:.4f}")
    else:
        print("WARNING: Face dataset not found. Skipping face model training.")
        return
    
    # Train Voice Model
    if os.path.exists(voice_model_path) and not force_training:
        print("\n" + "-"*40)
        print("Voice model already exists. Skipping training.")
        print("-"*40)
    elif voice_data_exists:
        if force_training and os.path.exists(voice_model_path):
            print("\n" + "-"*40)
            print("Force training enabled. Retraining Voice model...")
            print("-"*40)
        else:
            print("\n" + "-"*40)
            print("Training Voice Stress LSTM...")
            print("-"*40)
        
        voice_model, voice_acc = train_voice_model(
            data_dir=voice_data_dir,
            model_save_path=voice_model_path,
            epochs=50
        )
        print(f"Voice model training complete! Accuracy: {voice_acc:.4f}")
    else:
        print("WARNING: Voice dataset not found. Skipping voice model training.")
        return
    
    # Train Fusion Model
    if os.path.exists(fusion_model_path) and not force_training:
        print("\n" + "-"*40)
        print("Fusion model already exists. Skipping training.")
        print("-"*40)
    elif os.path.exists(face_model_path) and os.path.exists(voice_model_path):
        if force_training and os.path.exists(fusion_model_path):
            print("\n" + "-"*40)
            print("Force training enabled. Retraining Fusion model...")
            print("-"*40)
        else:
            print("\n" + "-"*40)
            print("Training Multimodal Fusion Model...")
            print("-"*40)
        
        try:
            fusion_model, fusion_acc = train_fusion_model(
                face_model_path=face_model_path,
                voice_model_path=voice_model_path,
                face_data_dir=face_data_dir,
                voice_data_dir=voice_data_dir,
                save_path=fusion_model_path
            )
            print(f"Fusion model training complete! Accuracy: {fusion_acc:.4f}")
        except Exception as e:
            print(f"Fusion model training failed: {e}")
            print("But face and voice models are trained and ready to use.")
    else:
        print("WARNING: Face or Voice model not found. Cannot train fusion model.")

def create_model_info():
    """Create model information file"""
    model_info = """
# Trained Models Information

## Face Stress CNN
- **Architecture**: Lightweight CNN with 4 conv layers
- **Input**: 48x48 grayscale facial images
- **Output**: 3 stress levels (Low, Medium, High)
- **Dataset**: FER-2013 (emotion -> stress mapping)
- **Features**: Batch normalization, dropout, adaptive pooling

## Voice Stress LSTM
- **Architecture**: Bidirectional LSTM with attention
- **Input**: Audio features (MFCC, pitch, energy, spectral)
- **Output**: 3 stress levels (Low, Medium, High)
- **Dataset**: RAVDESS (emotion -> stress mapping)
- **Features**: Attention mechanism, gradient clipping

## Multimodal Fusion Model
- **Architecture**: Feature concatenation + fully connected layers
- **Input**: Face features (64-dim) + Voice features (32-dim)
- **Output**: Stress level + Confidence score (0-1)
- **Training**: Uses pre-trained face and voice models
- **Features**: Dual-head architecture for stress and confidence

## Usage in Real-time System
1. Extract face features using Face CNN
2. Extract voice features using Voice LSTM
3. Combine features in Fusion Model
4. Output stress level and confidence score
5. Apply temporal smoothing for stability

## Model Files
- `face_stress_model.pth`: Face CNN weights
- `voice_stress_model.pth`: Voice LSTM weights  
- `fusion_model.pth`: Fusion model weights
- `scaler.pkl`: Audio feature scaler (in RAVDESS preprocessed folder)
"""
    
    with open("../../models/trained/MODEL_INFO.md", "w") as f:
        f.write(model_info)
    
    print("Model information saved to MODEL_INFO.md")

def main():
    parser = argparse.ArgumentParser(description="Train Interview Stress Analysis Models")
    parser.add_argument("--fer2013_csv", default="../../datasets/fer2013/fer2013.csv",
                       help="Path to FER-2013 CSV file")
    parser.add_argument("--ravdess_dir", default="../../datasets/ravdess/audio_speech_actors_01-24",
                       help="Path to RAVDESS audio directory")
    parser.add_argument("--skip_preprocessing", action="store_true",
                       help="Skip dataset preprocessing")
    parser.add_argument("--force_preprocessing", action="store_true",
                       help="Force reprocessing even if preprocessed data exists")
    parser.add_argument("--force_training", action="store_true",
                       help="Force retraining even if trained models exist")
    parser.add_argument("--preprocessing_only", action="store_true",
                       help="Only run preprocessing, skip training")
    
    args = parser.parse_args()
    
    print("="*60)
    print("INTERVIEW STRESS ANALYSIS - MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    # Preprocessing
    if not args.skip_preprocessing:
        preprocess_datasets(args.fer2013_csv, args.ravdess_dir, args.force_preprocessing)
    
    if args.preprocessing_only:
        print("Preprocessing complete. Exiting...")
        return
    
    # Training
    train_models(args.force_training)
    
    # Create model info
    create_model_info()
    
    print("="*60)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*60)
    print("Trained models saved in: models/trained/")
    print("Ready for real-time inference!")

if __name__ == "__main__":
    main()