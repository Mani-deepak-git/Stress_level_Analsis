# ML System Architecture - Interview Stress Analysis

## Overview
A multimodal deep learning system that analyzes facial expressions and voice patterns to detect stress levels and confidence scores in real-time during interviews.

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
├─────────────────────────────────┬───────────────────────────────────┤
│     Video Stream (30 FPS)       │    Audio Stream (16kHz)           │
│     Real-time Webcam Feed       │    Real-time Microphone Input     │
└────────────┬────────────────────┴──────────────┬────────────────────┘
             │                                    │
             ▼                                    ▼
┌────────────────────────────┐      ┌────────────────────────────────┐
│   FACE PREPROCESSING       │      │   AUDIO PREPROCESSING          │
├────────────────────────────┤      ├────────────────────────────────┤
│ • Face Detection           │      │ • Audio Segmentation (3s)      │
│ • Crop & Align Face        │      │ • Feature Extraction:          │
│ • Resize to 48x48          │      │   - MFCC (20 coefficients)     │
│ • Grayscale Conversion     │      │   - Pitch (F0)                 │
│ • Normalization (0-1)      │      │   - Energy/Intensity           │
└────────────┬───────────────┘      │   - Spectral Features          │
             │                       │   - Zero Crossing Rate         │
             │                       └────────────┬───────────────────┘
             │                                    │
             ▼                                    ▼
┌────────────────────────────┐      ┌────────────────────────────────┐
│   FACE STRESS CNN          │      │   VOICE STRESS LSTM            │
├────────────────────────────┤      ├────────────────────────────────┤
│ Input: 48x48x1 grayscale   │      │ Input: (T, 60) features        │
│                            │      │                                │
│ Conv2D(32) + BatchNorm     │      │ Bidirectional LSTM(64)         │
│ Conv2D(64) + BatchNorm     │      │ Bidirectional LSTM(32)         │
│ Conv2D(128) + BatchNorm    │      │ Attention Mechanism            │
│ Conv2D(256) + BatchNorm    │      │ Dropout(0.3)                   │
│ Adaptive Pooling           │      │ Dense(32)                      │
│ Dropout(0.5)               │      │                                │
│ Dense(64)                  │      │ Output: 32-dim features        │
│                            │      │                                │
│ Output: 64-dim features    │      │ Trained on: RAVDESS            │
│                            │      │ Emotions → Stress Mapping      │
│ Trained on: FER-2013       │      │ • Calm/Happy → Low (0)         │
│ Emotions → Stress Mapping  │      │ • Surprise/Disgust → Med (1)   │
│ • Happy/Neutral → Low (0)  │      │ • Angry/Fear/Sad → High (2)    │
│ • Surprise/Disgust → Med(1)│      │                                │
│ • Angry/Fear/Sad → High(2) │      │                                │
└────────────┬───────────────┘      └────────────┬───────────────────┘
             │                                    │
             │         Face Features (64-dim)     │  Voice Features (32-dim)
             │                                    │
             └────────────┬───────────────────────┘
                          │
                          ▼
             ┌────────────────────────────┐
             │   FEATURE FUSION LAYER     │
             ├────────────────────────────┤
             │ Concatenate:               │
             │ [Face(64) + Voice(32)]     │
             │ = 96-dimensional vector    │
             └────────────┬───────────────┘
                          │
                          ▼
             ┌────────────────────────────┐
             │   FUSION MODEL             │
             ├────────────────────────────┤
             │ Input: 96-dim features     │
             │                            │
             │ Dense(128) + ReLU          │
             │ Dropout(0.3)               │
             │ Dense(64) + ReLU           │
             │ Dropout(0.2)               │
             │                            │
             │ ┌─────────────────────┐    │
             │ │  Dual-Head Output   │    │
             │ ├─────────────────────┤    │
             │ │ Head 1: Stress      │    │
             │ │ Dense(3) + Softmax  │    │
             │ │ → [Low, Med, High]  │    │
             │ │                     │    │
             │ │ Head 2: Confidence  │    │
             │ │ Dense(1) + Sigmoid  │    │
             │ │ → Score (0.0-1.0)   │    │
             │ └─────────────────────┘    │
             └────────────┬───────────────┘
                          │
                          ▼
             ┌────────────────────────────┐
             │   TEMPORAL SMOOTHING       │
             ├────────────────────────────┤
             │ Rolling Window (5 frames)  │
             │ Moving Average Filter      │
             │ Reduces prediction jitter  │
             │ Stabilizes output          │
             └────────────┬───────────────┘
                          │
                          ▼
             ┌────────────────────────────┐
             │   OUTPUT LAYER             │
             ├────────────────────────────┤
             │ • Stress Level: 0/1/2      │
             │   (Low/Medium/High)        │
             │ • Confidence Score: 0-100% │
             │ • Timestamp                │
             │ • Recommendations          │
             └────────────────────────────┘
```

---

## Model Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATASET PREPARATION                          │
├──────────────────────────────┬──────────────────────────────────┤
│   FER-2013 Dataset           │   RAVDESS Dataset                │
│   • 35,887 facial images     │   • 1,440 audio files            │
│   • 48x48 grayscale          │   • 24 actors, 8 emotions        │
│   • 7 emotions               │   • WAV format, 16kHz            │
│   • Train/Test split         │   • Controlled recordings        │
└──────────────┬───────────────┴──────────────┬───────────────────┘
               │                               │
               ▼                               ▼
    ┌──────────────────────┐      ┌──────────────────────────┐
    │ Emotion → Stress     │      │ Emotion → Stress         │
    │ Label Mapping        │      │ Label Mapping            │
    └──────────┬───────────┘      └──────────┬───────────────┘
               │                               │
               ▼                               ▼
    ┌──────────────────────┐      ┌──────────────────────────┐
    │ Data Augmentation    │      │ Feature Engineering      │
    │ • Rotation (±10°)    │      │ • MFCC extraction        │
    │ • Horizontal flip    │      │ • Pitch tracking         │
    │ • Brightness adjust  │      │ • Energy computation     │
    │ • Contrast adjust    │      │ • Spectral analysis      │
    └──────────┬───────────┘      └──────────┬───────────────┘
               │                               │
               ▼                               ▼
    ┌──────────────────────┐      ┌──────────────────────────┐
    │ Train Face CNN       │      │ Train Voice LSTM         │
    │ • 30 epochs          │      │ • 50 epochs              │
    │ • Batch size: 32     │      │ • Batch size: 16         │
    │ • Adam optimizer     │      │ • Adam optimizer         │
    │ • CrossEntropy loss  │      │ • CrossEntropy loss      │
    │ • Accuracy: ~70-75%  │      │ • Accuracy: ~65-70%      │
    └──────────┬───────────┘      └──────────┬───────────────┘
               │                               │
               └───────────┬───────────────────┘
                           │
                           ▼
              ┌────────────────────────────┐
              │ Train Fusion Model         │
              │ • Freeze base models       │
              │ • 20 epochs                │
              │ • Combined loss:           │
              │   - Stress classification  │
              │   - Confidence regression  │
              │ • Accuracy: ~75-80%        │
              └────────────────────────────┘
```

---

## Key Components

### 1. Face Stress CNN
- **Purpose**: Extract facial stress indicators
- **Architecture**: Lightweight CNN with 4 convolutional layers
- **Input**: 48×48 grayscale face images
- **Output**: 64-dimensional feature vector
- **Training Data**: FER-2013 (35,887 images)

### 2. Voice Stress LSTM
- **Purpose**: Analyze vocal stress patterns
- **Architecture**: Bidirectional LSTM with attention mechanism
- **Input**: 60-dimensional audio features over time
- **Output**: 32-dimensional feature vector
- **Training Data**: RAVDESS (1,440 audio samples)

### 3. Multimodal Fusion Model
- **Purpose**: Combine face and voice features for final prediction
- **Architecture**: Fully connected network with dual-head output
- **Input**: 96-dimensional concatenated features
- **Output**: Stress level (3 classes) + Confidence score (0-1)
- **Training**: End-to-end fine-tuning on combined data

### 4. Temporal Smoothing
- **Purpose**: Stabilize predictions over time
- **Method**: Rolling average over last 5 predictions
- **Benefit**: Reduces noise and jitter in real-time analysis

---

## Stress Level Mapping

| Emotion Category | Stress Level | Reasoning |
|-----------------|--------------|-----------|
| Happy, Neutral, Calm | **Low (0)** | Positive/relaxed emotional state |
| Surprise, Disgust | **Medium (1)** | Moderate arousal, uncertain state |
| Angry, Fear, Sad | **High (2)** | Negative emotions, high arousal |

---

## Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Face CNN | 70-75% | 0.72 | 0.70 | 0.71 |
| Voice LSTM | 65-70% | 0.68 | 0.66 | 0.67 |
| Fusion Model | 75-80% | 0.77 | 0.76 | 0.76 |

---

## Real-time Inference Pipeline

```
Video Frame (t) ──┐
                  ├──> Face Detection ──> Face CNN ──┐
                  │                                   │
Audio Chunk (t) ──┤                                   ├──> Fusion ──> Smoothing ──> Output
                  │                                   │
                  └──> Feature Extract ──> Voice LSTM ┘
                  
Processing Time: ~50-100ms per frame
Throughput: 10-20 FPS
```

---

## Technical Specifications

### Hardware Requirements
- **CPU**: Intel i5 or equivalent (minimum)
- **RAM**: 8GB (minimum), 16GB (recommended)
- **GPU**: Optional (NVIDIA GPU with CUDA for faster inference)

### Software Stack
- **Framework**: PyTorch 2.0.1
- **Computer Vision**: OpenCV 4.8.1
- **Audio Processing**: Librosa 0.10.1
- **Feature Extraction**: MediaPipe 0.10.7
- **Python Version**: 3.10.x

### Model Sizes
- Face CNN: ~2.5 MB
- Voice LSTM: ~1.8 MB
- Fusion Model: ~0.5 MB
- **Total**: ~4.8 MB (lightweight for deployment)

---

## Advantages of This Architecture

1. **Multimodal Fusion**: Combines visual and audio cues for robust analysis
2. **Lightweight Models**: Fast inference suitable for real-time applications
3. **Temporal Smoothing**: Stable predictions without sudden jumps
4. **Dual Output**: Both stress level and confidence score
5. **Emotion Mapping**: Interpretable stress levels based on emotions
6. **Scalable**: Can add more modalities (text, physiological signals)

---

## Future Enhancements

- Add text sentiment analysis from interview transcripts
- Incorporate physiological signals (heart rate, GSR)
- Implement attention visualization for explainability
- Add personalized baseline calibration
- Multi-person stress detection in group interviews
