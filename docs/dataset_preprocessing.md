# Dataset Preprocessing Guide

## FER-2013 (Facial Expression Recognition)
**Purpose**: Train CNN to detect facial stress indicators
**Source**: Kaggle FER-2013 dataset
**Format**: 48x48 grayscale images
**Labels**: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

### Stress Mapping Strategy:
- **High Stress**: Angry, Fear, Sad (labels 0, 2, 4)
- **Low Stress**: Happy, Neutral (labels 3, 6)
- **Medium Stress**: Surprise, Disgust (labels 1, 5)

### Preprocessing Steps:
1. Load CSV data (pixel values as strings)
2. Convert pixel strings to 48x48 numpy arrays
3. Normalize pixel values (0-255 → 0-1)
4. Map emotion labels to stress levels
5. Split: 80% train, 20% validation
6. Apply data augmentation (rotation, flip, brightness)

## RAVDESS (Audio Emotion Recognition)
**Purpose**: Train LSTM to detect voice stress indicators
**Source**: Kaggle RAVDESS dataset
**Format**: WAV audio files (48kHz, 16-bit)
**Labels**: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised

### Stress Mapping Strategy:
- **High Stress**: Angry, Fearful, Sad (05, 06, 04)
- **Low Stress**: Calm, Neutral, Happy (02, 01, 03)
- **Medium Stress**: Disgust, Surprised (07, 08)

### Audio Feature Extraction:
1. **MFCC**: 13 coefficients (spectral characteristics)
2. **Pitch**: Fundamental frequency (F0)
3. **Energy**: RMS energy levels
4. **Spectral Features**: Centroid, rolloff, zero-crossing rate
5. **Temporal Features**: Duration, pause patterns

### Preprocessing Steps:
1. Load WAV files using librosa
2. Resample to 16kHz for consistency
3. Extract MFCC features (13 coefficients)
4. Extract pitch using librosa.piptrack()
5. Calculate RMS energy
6. Normalize features using StandardScaler
7. Create fixed-length sequences (padding/truncating)
8. Map emotion labels to stress levels

## Model Training Strategy:
1. **Face CNN**: Learn facial tension patterns
2. **Voice LSTM**: Learn speech stress patterns  
3. **Fusion Model**: Combine both modalities for final prediction