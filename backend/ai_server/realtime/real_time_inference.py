"""
Real-Time Inference Engine
Loads trained LSTM model and performs real-time inference
Why: Uses existing trained model WITHOUT modification
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.voice_model import VoiceStressLSTM

class RealTimeInference:
    """
    Real-time inference using trained LSTM model
    Why: Loads and uses existing model without any modifications
    """
    
    def __init__(self, model_path, device='cpu'):
        """
        Args:
            model_path: Path to trained voice_stress_model.pth
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model = None
        self.model_info = None
        
        # Load model
        self._load_model(model_path)
        
        # Stress level mapping (matching training)
        self.stress_labels = ['Low Stress', 'Medium Stress', 'High Stress']
        
        print(f"Real-time inference initialized on {device}")
    
    def _load_model(self, model_path):
        """
        Load trained model
        Why: Reuses existing trained weights without modification
        """
        # Resolve absolute path
        if not os.path.isabs(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.normpath(os.path.join(script_dir, model_path))
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_info = checkpoint
        
        # Initialize model with saved architecture
        self.model = VoiceStressLSTM(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            num_classes=checkpoint['num_classes']
        )
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"Model loaded: {checkpoint['model_architecture']}")
        print(f"Input size: {checkpoint['input_size']}")
        print(f"Classes: {checkpoint['num_classes']}")
    
    def predict(self, features):
        """
        Perform inference on feature vector
        Why: Converts features to stress prediction and confidence
        
        Args:
            features: Feature vector from LiveFeatureExtractor
            
        Returns:
            dict with stress_level, confidence, probabilities
        """
        if features is None:
            return None
        
        try:
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Inference (no gradient computation)
            with torch.no_grad():
                logits = self.model(features_tensor)
                probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
                predicted_class = np.argmax(probabilities)
            
            # Convert to confidence score (0-100)
            confidence = self._calculate_confidence(predicted_class, probabilities)
            
            return {
                'stress_level': self.stress_labels[predicted_class],
                'stress_class': int(predicted_class),
                'confidence': float(confidence),
                'probabilities': probabilities.tolist(),
                'raw_logits': logits.cpu().numpy()[0].tolist()
            }
            
        except Exception as e:
            print(f"Inference error: {e}")
            return None
    
    def _calculate_confidence(self, predicted_class, probabilities):
        """
        Calculate confidence score (0-100) from stress prediction
        Why: Converts stress level to confidence metric
        
        Mapping (voice-based confidence estimation):
        - Low Stress (class 0)    → High Confidence (80-100)
        - Medium Stress (class 1) → Medium Confidence (40-79)
        - High Stress (class 2)   → Low Confidence (0-39)
        
        Note: This is a PROXY measure based on emotional intensity,
        not direct psychological confidence measurement
        """
        # Get probability of predicted class (certainty)
        certainty = probabilities[predicted_class]
        
        if predicted_class == 0:  # Low Stress
            # Map to 80-100 range
            base_confidence = 80
            confidence = base_confidence + (certainty * 20)
        elif predicted_class == 1:  # Medium Stress
            # Map to 40-79 range
            base_confidence = 40
            confidence = base_confidence + (certainty * 39)
        else:  # High Stress (class 2)
            # Map to 0-39 range
            confidence = certainty * 39
        
        return np.clip(confidence, 0, 100)
    
    def get_model_info(self):
        """Get model architecture information"""
        return {
            'architecture': self.model_info.get('model_architecture', 'Unknown'),
            'input_size': self.model_info.get('input_size', 0),
            'hidden_size': self.model_info.get('hidden_size', 0),
            'num_layers': self.model_info.get('num_layers', 0),
            'num_classes': self.model_info.get('num_classes', 0)
        }


if __name__ == "__main__":
    # Test inference
    print("Testing real-time inference...")
    
    # Resolve absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.normpath(os.path.join(script_dir, "../../models/trained/voice_stress_model.pth"))
    
    if os.path.exists(model_path):
        inference = RealTimeInference(model_path)
        
        # Test with dummy features
        dummy_features = np.random.randn(60)  # 60-dim feature vector
        result = inference.predict(dummy_features)
        
        if result:
            print(f"Stress Level: {result['stress_level']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Probabilities: {result['probabilities']}")
        
        print("Model info:", inference.get_model_info())
    else:
        print(f"Model not found: {model_path}")
        print("Please train the model first")
    
    print("Test complete")
