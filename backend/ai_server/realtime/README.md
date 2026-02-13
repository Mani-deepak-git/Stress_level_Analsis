# Real-Time Voice Confidence Analysis System

## 📋 Overview

This module provides **real-time voice-based confidence estimation** using your existing trained LSTM model. It captures live microphone audio, extracts features, performs inference, and streams confidence data to the UI for heartbeat-style visualization.

**Important**: This is a **proxy measure** based on emotional intensity and stress patterns, NOT direct psychological confidence measurement.

---

## 🏗️ Architecture

```
Microphone → Audio Capture → Feature Extraction → Model Inference → Smoothing → WebSocket → UI
```

### Components

1. **live_audio_capture.py**: Captures audio from microphone with sliding window
2. **live_feature_extractor.py**: Extracts features matching RAVDESS preprocessing
3. **real_time_inference.py**: Loads trained model and performs inference
4. **confidence_smoother.py**: Applies temporal smoothing (EMA/Kalman)
5. **realtime_stream_server.py**: WebSocket server for frontend streaming

---

## 🔒 What Was NOT Modified

✅ **Model architecture** - Unchanged  
✅ **Training logic** - Unchanged  
✅ **Loss functions** - Unchanged  
✅ **Dataset preprocessing** - Unchanged  

Only **new modules** were added for real-time processing.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Activate your Python environment
ai_env\Scripts\activate

# Install additional dependency
pip install sounddevice==0.4.6
```

### 2. Verify Model Exists

Ensure your trained model is at:
```
models/trained/voice_stress_model.pth
```

### 3. Start Real-Time Server

```bash
cd backend\ai_server\realtime
python realtime_stream_server.py
```

Server starts on: `http://localhost:8002`  
WebSocket endpoint: `ws://localhost:8002/ws/voice-confidence`

### 4. Test Individual Components

```bash
# Test audio capture
python live_audio_capture.py

# Test feature extraction
python live_feature_extractor.py

# Test inference
python real_time_inference.py

# Test smoother
python confidence_smoother.py
```

---

## 📡 WebSocket API

### Connect

```javascript
const ws = new WebSocket('ws://localhost:8002/ws/voice-confidence');
```

### Receive Data (every ~300ms)

```json
{
  "timestamp": 1700000000,
  "confidence": 72.4,
  "stress_level": "Low Stress",
  "stress_class": 0,
  "probabilities": [0.8, 0.15, 0.05],
  "raw_confidence": 75.2
}
```

### Send Commands

```javascript
// Reset smoother
ws.send("reset");
```

---

## 🎯 Confidence Mapping

The system maps stress predictions to confidence scores:

| Stress Level | Confidence Range | Interpretation |
|--------------|------------------|----------------|
| **Low Stress** (class 0) | 80-100 | High confidence, calm voice |
| **Medium Stress** (class 1) | 40-79 | Moderate confidence |
| **High Stress** (class 2) | 0-39 | Low confidence, stressed voice |

**Formula**:
```python
if class == 0:  # Low Stress
    confidence = 80 + (probability * 20)
elif class == 1:  # Medium Stress
    confidence = 40 + (probability * 39)
else:  # High Stress
    confidence = probability * 39
```

---

## ⚙️ Configuration

Edit `realtime_stream_server.py`:

```python
CONFIG = {
    'sample_rate': 22050,        # Must match training
    'window_duration': 2.0,      # Audio window size (seconds)
    'overlap': 0.5,              # 50% overlap for sliding window
    'update_interval': 0.3,      # 300ms updates (heartbeat rate)
    'model_path': '../../models/trained/voice_stress_model.pth',
    'scaler_path': '../../datasets/ravdess/preprocessed/scaler.pkl'
}
```

### Smoothing Configuration

Edit `confidence_smoother.py`:

```python
smoother = ConfidenceSmoother(
    window_size=5,    # Number of samples to average
    method='ema',     # 'ma' or 'ema'
    alpha=0.3         # EMA smoothing factor (lower = smoother)
)
```

---

## 🎨 UI Integration Example

### React Component

```javascript
import React, { useEffect, useState } from 'react';

function VoiceConfidenceHeartbeat() {
  const [confidence, setConfidence] = useState(50);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8002/ws/voice-confidence');
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.confidence !== undefined) {
        setConfidence(data.confidence);
        setHistory(prev => [...prev.slice(-50), {
          time: data.timestamp,
          value: data.confidence
        }]);
      }
    };

    return () => ws.close();
  }, []);

  return (
    <div>
      <h2>Voice Confidence: {confidence.toFixed(1)}%</h2>
      {/* Add line chart here using history data */}
    </div>
  );
}
```

---

## 🔧 Troubleshooting

### Issue: No audio captured

**Solution**:
- Check microphone permissions
- Verify microphone is default input device
- Try alternative: Install `pyaudio` instead of `sounddevice`

### Issue: Feature dimension mismatch

**Solution**:
- Ensure scaler.pkl exists from training
- Verify feature extraction matches training exactly
- Check model input_size matches feature dimension

### Issue: Model not found

**Solution**:
- Train the voice model first: `python train_models.py`
- Verify path: `models/trained/voice_stress_model.pth`

### Issue: Confidence fluctuates too much

**Solution**:
- Increase smoother window_size (e.g., 10)
- Decrease alpha for EMA (e.g., 0.2)
- Use Kalman filter instead

---

## 📊 Performance

- **Latency**: ~300-500ms (audio window + processing)
- **Update Rate**: 3-4 times per second
- **CPU Usage**: Low (CPU-only inference)
- **Memory**: ~200MB

---

## 🧪 Testing

### Test Audio Capture
```bash
python live_audio_capture.py
# Should capture 5 seconds of audio
```

### Test Feature Extraction
```bash
python live_feature_extractor.py
# Should output 60-dimensional feature vector
```

### Test Inference
```bash
python real_time_inference.py
# Should load model and predict on dummy data
```

### Test Full Pipeline
```bash
python realtime_stream_server.py
# Connect via WebSocket and observe data stream
```

---

## 📝 Important Notes

### Confidence Interpretation

This system provides **voice-based confidence estimation** as a proxy measure:

✅ **What it measures**:
- Vocal stress patterns
- Emotional intensity
- Speech characteristics

❌ **What it does NOT measure**:
- True psychological confidence
- Cognitive certainty
- Personality traits

### Ethical Considerations

- Inform users that voice is being analyzed
- Explain it's a proxy measure, not psychological assessment
- Do not use for high-stakes decisions
- Provide opt-out mechanism

---

## 🔄 Integration with Existing System

### Add to main FastAPI server

```python
# In main.py
from realtime.realtime_stream_server import app as realtime_app

# Mount as sub-application
app.mount("/realtime", realtime_app)
```

### Or run as separate service

Keep it as standalone server on port 8002 (recommended for isolation)

---

## 📚 References

- **RAVDESS Dataset**: Livingstone & Russo (2018)
- **LSTM Architecture**: Your trained model
- **Feature Extraction**: librosa library
- **Smoothing**: Exponential Moving Average

---

## ✅ Checklist

Before deploying:

- [ ] Trained model exists
- [ ] Scaler.pkl exists (from training)
- [ ] Microphone permissions granted
- [ ] sounddevice installed
- [ ] WebSocket port 8002 available
- [ ] Frontend configured to connect
- [ ] Users informed about voice analysis

---

## 🎯 Expected Output

When running, you should see:

```
Audio Capture initialized: 22050Hz, 2.0s window, 50.0% overlap
Loaded scaler from ../../datasets/ravdess/preprocessed/scaler.pkl
Model loaded: VoiceStressLSTM
Input size: 60
Classes: 3
Smoother initialized: method=ema, window=5, alpha=0.3
All components initialized successfully
Audio capture started
Real-time voice confidence server started
WebSocket endpoint: ws://localhost:8002/ws/voice-confidence
Update interval: 300.0ms
```

Then continuous data stream to connected clients.

---

**Built to work with your existing trained model - no modifications required!** 🚀
