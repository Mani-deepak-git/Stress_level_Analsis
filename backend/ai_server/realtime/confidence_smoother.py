"""
Confidence Smoother
Applies temporal smoothing to reduce prediction fluctuations
Why: Raw predictions fluctuate too much for good UX
"""

import numpy as np
from collections import deque

class ConfidenceSmoother:
    """
    Smooths confidence values over time
    Why: Creates stable, visually pleasing confidence curve
    """
    
    def __init__(self, window_size=5, method='ema', alpha=0.3):
        """
        Args:
            window_size: Number of samples for moving average
            method: 'ma' (moving average) or 'ema' (exponential moving average)
            alpha: Smoothing factor for EMA (0-1, lower = smoother)
        """
        self.window_size = window_size
        self.method = method
        self.alpha = alpha
        
        # History buffers
        self.confidence_history = deque(maxlen=window_size)
        self.stress_class_history = deque(maxlen=window_size)
        
        # EMA state
        self.ema_confidence = None
        
        print(f"Smoother initialized: method={method}, window={window_size}, alpha={alpha}")
    
    def smooth(self, confidence, stress_class):
        """
        Apply smoothing to confidence value
        Why: Reduces jitter and creates smooth transitions
        
        Args:
            confidence: Raw confidence value (0-100)
            stress_class: Predicted stress class (0, 1, 2)
            
        Returns:
            Smoothed confidence value
        """
        # Add to history
        self.confidence_history.append(confidence)
        self.stress_class_history.append(stress_class)
        
        if self.method == 'ma':
            # Moving Average
            smoothed = self._moving_average()
        elif self.method == 'ema':
            # Exponential Moving Average
            smoothed = self._exponential_moving_average(confidence)
        else:
            smoothed = confidence
        
        return np.clip(smoothed, 0, 100)
    
    def _moving_average(self):
        """
        Simple moving average
        Why: Averages last N predictions for stability
        """
        if len(self.confidence_history) == 0:
            return 50.0  # Default
        
        return np.mean(self.confidence_history)
    
    def _exponential_moving_average(self, current_value):
        """
        Exponential moving average
        Why: Gives more weight to recent values while smoothing
        """
        if self.ema_confidence is None:
            self.ema_confidence = current_value
            return current_value
        
        # EMA formula: EMA_t = alpha * value_t + (1 - alpha) * EMA_{t-1}
        self.ema_confidence = self.alpha * current_value + (1 - self.alpha) * self.ema_confidence
        
        return self.ema_confidence
    
    def get_smoothed_stress_level(self):
        """
        Get most common stress level from recent history
        Why: Prevents rapid stress level switching
        """
        if len(self.stress_class_history) == 0:
            return 'Medium Stress'
        
        # Mode (most common)
        most_common = max(set(self.stress_class_history), key=list(self.stress_class_history).count)
        
        labels = ['Low Stress', 'Medium Stress', 'High Stress']
        return labels[most_common]
    
    def reset(self):
        """Reset smoother state"""
        self.confidence_history.clear()
        self.stress_class_history.clear()
        self.ema_confidence = None
        print("Smoother reset")


class KalmanFilter:
    """
    Optional: Kalman filter for advanced smoothing
    Why: Provides optimal smoothing with noise estimation
    """
    
    def __init__(self, process_variance=1e-5, measurement_variance=1e-2):
        """
        Args:
            process_variance: How much we expect the value to change
            measurement_variance: How noisy our measurements are
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        
        # State
        self.estimate = None
        self.estimate_error = 1.0
    
    def update(self, measurement):
        """
        Update Kalman filter with new measurement
        Why: Provides statistically optimal smoothing
        """
        if self.estimate is None:
            self.estimate = measurement
            return measurement
        
        # Prediction
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance
        
        # Update
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error
        
        return self.estimate
    
    def reset(self):
        """Reset filter state"""
        self.estimate = None
        self.estimate_error = 1.0


if __name__ == "__main__":
    # Test smoother
    print("Testing confidence smoother...")
    
    # Simulate noisy confidence values
    np.random.seed(42)
    true_confidence = 70
    noisy_values = true_confidence + np.random.randn(20) * 10
    
    # Test Moving Average
    print("\n--- Moving Average ---")
    smoother_ma = ConfidenceSmoother(window_size=5, method='ma')
    for i, value in enumerate(noisy_values):
        smoothed = smoother_ma.smooth(value, stress_class=0)
        print(f"Step {i}: Raw={value:.2f}, Smoothed={smoothed:.2f}")
    
    # Test EMA
    print("\n--- Exponential Moving Average ---")
    smoother_ema = ConfidenceSmoother(window_size=5, method='ema', alpha=0.3)
    for i, value in enumerate(noisy_values):
        smoothed = smoother_ema.smooth(value, stress_class=0)
        print(f"Step {i}: Raw={value:.2f}, Smoothed={smoothed:.2f}")
    
    # Test Kalman Filter
    print("\n--- Kalman Filter ---")
    kalman = KalmanFilter()
    for i, value in enumerate(noisy_values):
        smoothed = kalman.update(value)
        print(f"Step {i}: Raw={value:.2f}, Smoothed={smoothed:.2f}")
    
    print("\nTest complete")
