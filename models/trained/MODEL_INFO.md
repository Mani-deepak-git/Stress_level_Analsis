
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
