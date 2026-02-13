# Dataset Setup Instructions

This document provides detailed instructions for downloading and setting up the required datasets for training the AI models.

## Required Datasets

### 1. FER-2013 (Facial Expression Recognition)
- **Purpose**: Train facial stress detection CNN
- **Source**: Kaggle
- **Size**: ~35,000 images (48x48 grayscale)
- **Format**: CSV file with pixel values

#### Download Steps:
1. Go to [Kaggle FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
2. Click "Download" (requires Kaggle account)
3. Extract the downloaded ZIP file
4. Copy `fer2013.csv` to `datasets/fer2013/fer2013.csv`

#### Expected Structure:
```
datasets/fer2013/
├── fer2013.csv
└── preprocessed/  (created during training)
```

### 2. RAVDESS (Audio Emotion Recognition)
- **Purpose**: Train voice stress detection LSTM
- **Source**: Kaggle
- **Size**: ~1,440 audio files
- **Format**: WAV files (48kHz, 16-bit)

#### Download Steps:
1. Go to [Kaggle RAVDESS Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
2. Click "Download" (requires Kaggle account)
3. Extract the downloaded ZIP file
4. Copy the `audio_speech_actors_01-24` folder to `datasets/ravdess/`

#### Expected Structure:
```
datasets/ravdess/
├── audio_speech_actors_01-24/
│   ├── Actor_01/
│   ├── Actor_02/
│   └── ... (24 actors total)
└── preprocessed/  (created during training)
```

## Alternative: Pre-trained Models

If you don't want to download datasets and train from scratch, you can use pre-trained models:

### Option 1: Skip Training (Use Dummy Models)
The system will work with randomly initialized models for demonstration purposes:
```bash
# Just run the system without training
start_full_system.bat
```

### Option 2: Download Pre-trained Models
If pre-trained models are available:
1. Download model files
2. Place in `models/trained/` directory:
   - `face_stress_model.pth`
   - `voice_stress_model.pth`
   - `fusion_model.pth`

## Dataset Information

### FER-2013 Details
- **Classes**: 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Stress Mapping**:
  - High Stress: Angry, Fear, Sad
  - Medium Stress: Disgust, Surprise
  - Low Stress: Happy, Neutral
- **Image Size**: 48x48 pixels, grayscale
- **Total Images**: ~35,887

### RAVDESS Details
- **Classes**: 8 emotions (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised)
- **Stress Mapping**:
  - High Stress: Angry, Fearful, Sad
  - Medium Stress: Disgust, Surprised
  - Low Stress: Neutral, Calm, Happy
- **Audio Format**: WAV, 48kHz, 16-bit
- **Total Files**: 1,440 (24 actors × 60 recordings each)

## Preprocessing Overview

### Face Data Preprocessing
1. Load CSV with pixel values
2. Convert strings to 48x48 numpy arrays
3. Normalize pixel values (0-255 → 0-1)
4. Map emotion labels to stress levels
5. Apply data augmentation (rotation, flip, brightness)
6. Split into train/validation sets

### Audio Data Preprocessing
1. Load WAV files using librosa
2. Extract features:
   - MFCC (13 coefficients)
   - Pitch (F0)
   - Energy (RMS)
   - Spectral features
   - Zero-crossing rate
   - Chroma features
3. Normalize features using StandardScaler
4. Map emotion labels to stress levels
5. Split into train/validation sets

## Training Process

### Automated Training
```bash
# Train all models automatically
cd backend\ai_server
python train_models.py
```

### Manual Training Steps
```bash
# 1. Preprocess datasets only
python train_models.py --preprocessing_only

# 2. Train with custom paths
python train_models.py --fer2013_csv "path/to/fer2013.csv" --ravdess_dir "path/to/ravdess"

# 3. Skip preprocessing if already done
python train_models.py --skip_preprocessing
```

## Verification

After setup, verify your datasets:

### Check FER-2013
```bash
cd backend\ai_server\preprocessing
python fer2013_preprocessor.py
```

### Check RAVDESS
```bash
cd backend\ai_server\preprocessing
python ravdess_preprocessor.py
```

## Troubleshooting

### Common Issues

**1. CSV File Not Found**
```
FileNotFoundError: fer2013.csv not found
```
**Solution**: Ensure `fer2013.csv` is in `datasets/fer2013/` directory

**2. Audio Directory Not Found**
```
FileNotFoundError: RAVDESS directory not found
```
**Solution**: Ensure `audio_speech_actors_01-24` folder is in `datasets/ravdess/`

**3. Kaggle Download Issues**
- Create free Kaggle account
- Verify email address
- Accept dataset terms and conditions

**4. Large File Sizes**
- FER-2013 CSV: ~95MB
- RAVDESS ZIP: ~1.2GB
- Ensure sufficient disk space

### Performance Notes

**Training Time Estimates** (CPU):
- Face Model: 30-60 minutes
- Voice Model: 45-90 minutes  
- Fusion Model: 15-30 minutes

**Memory Requirements**:
- RAM: 8GB minimum, 16GB recommended
- Disk Space: 5GB for datasets + models

## License Compliance

### FER-2013 License
- Academic use only
- Cite original paper if publishing results
- No commercial use without permission

### RAVDESS License
- Academic research license
- Cite dataset in publications
- Contact authors for commercial use

### Citation Requirements
```bibtex
@article{fer2013,
  title={Challenges in representation learning: A report on three machine learning contests},
  author={Goodfellow, Ian J and Erhan, Dumitru and Carrier, Pierre Luc and Courville, Aaron and Mirza, Mehdi and Hamner, Ben and Cukierski, Will and Tang, Yichuan and Thaler, David and Lee, Dong-Hyun and others},
  journal={arXiv preprint arXiv:1307.0414},
  year={2013}
}

@article{ravdess,
  title={The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English},
  author={Livingstone, Steven R and Russo, Frank A},
  journal={PloS one},
  volume={13},
  number={5},
  pages={e0196391},
  year={2018}
}
```