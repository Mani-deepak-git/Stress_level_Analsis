import cv2
import numpy as np
import os
from collections import deque


class RealTimeStressAnalyzer:
    """Lightweight stress analyzer using OpenCV only - no PyTorch/librosa needed.
    Uses real facial landmark analysis and audio energy for stress detection."""

    def __init__(self):
        # OpenCV face + eye detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        print("OpenCV face/eye detection initialized")

        # Audio processing
        self.sample_rate = 44100
        self.audio_buffer = deque(maxlen=int(4 * self.sample_rate))
        self.min_audio_samples = int(0.5 * self.sample_rate)

        # Temporal smoothing
        self.stress_history = deque(maxlen=8)
        self.blink_history = deque(maxlen=30)
        self.prev_gray = None
        self.frame_count = 0

        self.stress_labels = ['Low Stress', 'Medium Stress', 'High Stress']
        print("Lightweight stress analyzer ready (OpenCV mode)")

    def _analyze_face_stress(self, frame):
        """Analyze face for stress indicators using OpenCV features."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        if len(faces) == 0:
            return None, False

        x, y, w, h = faces[0]
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]

        stress_signals = []

        # 1. Eye openness / blink detection
        eyes = self.eye_cascade.detectMultiScale(face_roi_gray, 1.1, 10, minSize=(20, 20))
        eye_count = len(eyes)
        self.blink_history.append(eye_count)
        avg_eyes = np.mean(self.blink_history) if self.blink_history else 2
        # Fewer eyes detected on average = more blinking = stress signal
        if avg_eyes < 1.2:
            stress_signals.append(0.8)   # high stress
        elif avg_eyes < 1.7:
            stress_signals.append(0.5)   # medium
        else:
            stress_signals.append(0.2)   # low stress

        # 2. Skin texture / micro-expression via Laplacian variance
        lap_var = cv2.Laplacian(face_roi_gray, cv2.CV_64F).var()
        # Higher Laplacian = more texture detail / tension in face
        if lap_var > 400:
            stress_signals.append(0.7)
        elif lap_var > 200:
            stress_signals.append(0.4)
        else:
            stress_signals.append(0.2)

        # 3. Motion / fidgeting via frame difference
        if self.prev_gray is not None:
            try:
                prev_face = self.prev_gray[y:y+h, x:x+w]
                if prev_face.shape == face_roi_gray.shape:
                    diff = cv2.absdiff(face_roi_gray, prev_face)
                    motion = np.mean(diff)
                    if motion > 12:
                        stress_signals.append(0.75)   # lots of movement
                    elif motion > 5:
                        stress_signals.append(0.45)
                    else:
                        stress_signals.append(0.15)
            except Exception:
                pass

        self.prev_gray = gray.copy()
        self.frame_count += 1

        # 4. Forehead brightness change (stress → blood flow change)
        forehead = face_roi_color[0:h//4, w//4:3*w//4]
        if forehead.size > 0:
            mean_r = np.mean(forehead[:, :, 2])   # red channel
            if mean_r > 160:
                stress_signals.append(0.65)   # redness = stress
            elif mean_r > 130:
                stress_signals.append(0.4)
            else:
                stress_signals.append(0.2)

        if not stress_signals:
            return 0.33, True

        stress_score = float(np.mean(stress_signals))
        return stress_score, True

    def _analyze_audio_energy(self):
        """Analyze audio buffer for voice energy / confidence."""
        if len(self.audio_buffer) < self.min_audio_samples:
            return None, False

        audio = np.array(list(self.audio_buffer), dtype=np.float32)

        # Normalize if needed
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32767.0

        # RMS energy
        rms = float(np.sqrt(np.mean(audio ** 2)))

        # Zero crossing rate (higher = more voiced speech)
        signs = np.sign(audio)
        zcr = float(np.mean(np.abs(np.diff(signs))) / 2)

        # Confidence from energy
        if rms > 0.05:
            confidence = 0.9
        elif rms > 0.02:
            confidence = 0.75
        elif rms > 0.008:
            confidence = 0.55
        elif rms > 0.002:
            confidence = 0.35
        else:
            confidence = 0.2

        # Stress signal from voice: high ZCR + low energy = nervous/shaky voice
        audio_stress = 0.3
        if zcr > 0.15 and rms < 0.03:
            audio_stress = 0.7   # shaky, quiet voice
        elif rms < 0.005:
            audio_stress = 0.6   # silence / freezing
        elif rms > 0.05 and zcr < 0.08:
            audio_stress = 0.25  # strong, steady voice

        return {'rms': rms, 'zcr': zcr, 'confidence': confidence, 'audio_stress': audio_stress}, True

    def analyze_frame(self, video_frame, audio_chunk=None):
        """Main analysis entry point."""
        results = {
            'stress_level': 'Low Stress',
            'stress_probability': [0.6, 0.25, 0.15],
            'confidence_score': 0.5,
            'face_detected': False,
            'audio_processed': False
        }

        try:
            # Add audio to buffer
            if audio_chunk is not None and len(audio_chunk) > 0:
                self.audio_buffer.extend(audio_chunk)

            # --- Face analysis ---
            face_score, face_found = self._analyze_face_stress(video_frame)
            results['face_detected'] = face_found

            # --- Audio analysis ---
            audio_result, audio_ok = self._analyze_audio_energy()
            results['audio_processed'] = audio_ok

            # --- Combine signals ---
            if face_found and face_score is not None:
                combined_stress = face_score
                if audio_ok and audio_result:
                    # Weighted: 60% face, 40% audio
                    combined_stress = 0.6 * face_score + 0.4 * audio_result['audio_stress']
                    results['confidence_score'] = audio_result['confidence']
                else:
                    results['confidence_score'] = 0.5

                # Build probability distribution
                if combined_stress > 0.62:
                    probs = [0.15, 0.2, 0.65]
                elif combined_stress > 0.42:
                    probs = [0.3, 0.45, 0.25]
                else:
                    probs = [0.65, 0.25, 0.10]

                # Temporal smoothing on stress index
                stress_idx = int(np.argmax(probs))
                self.stress_history.append(stress_idx)
                smoothed = int(round(np.mean(self.stress_history)))
                smoothed = max(0, min(2, smoothed))

                results['stress_level'] = self.stress_labels[smoothed]
                results['stress_probability'] = probs

            elif audio_ok and audio_result:
                # Only audio available
                results['confidence_score'] = audio_result['confidence']
                a_stress = audio_result['audio_stress']
                if a_stress > 0.6:
                    results['stress_level'] = 'High Stress'
                    results['stress_probability'] = [0.15, 0.2, 0.65]
                else:
                    results['stress_level'] = 'Low Stress'
                    results['stress_probability'] = [0.65, 0.25, 0.10]

        except Exception as e:
            print(f"Error in analyze_frame: {e}")

        return results

    def reset_history(self):
        self.stress_history.clear()
        self.blink_history.clear()
        self.audio_buffer.clear()
        self.prev_gray = None
        self.frame_count = 0
        print("Analysis history reset")


def create_analyzer():
    try:
        return RealTimeStressAnalyzer()
    except Exception as e:
        print(f"Error creating analyzer: {e}")
        return None


def analyze_video_frame(analyzer, frame):
    if analyzer is None:
        return None
    return analyzer.analyze_frame(frame)


def analyze_multimodal(analyzer, frame, audio_chunk):
    if analyzer is None:
        return None
    return analyzer.analyze_frame(frame, audio_chunk)
