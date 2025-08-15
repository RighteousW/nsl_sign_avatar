import pickle
import numpy as np
import cv2
import sys
import os
import mediapipe as mp

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import FRAME_RATE, VIDEO_WIDTH, VIDEO_HEIGHT, MODEL_DIR, TRAINED_MODEL_DIR
from helper_functions import extract_3D_landmarks_from_frame, load_model


class GestureRecognizer:
    def __init__(self, model_path, scaler_path, label_encoder_path):
        """Initialize the gesture recognizer with model and preprocessing components"""

        # Load model and preprocessing components
        self.model = load_model(model_path)

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        # Configuration parameters (should match training)
        self.sequence_length = int(1 * FRAME_RATE)  # 1 second at current frame rate
        self.num_features = 176
        self.min_confidence = 0.7

        # Frame buffer for sequence processing
        self.frame_buffer = []
        self.min_inference_frames = 15
        self.max_inference_frames = 30

        print(f"Model loaded with sequence length: {self.sequence_length}")
        print(f"Classes: {list(self.label_encoder.classes_)}")

    def add_frame_landmarks(self, landmarks_data):
        """Add landmarks from a frame to the buffer"""
        # landmarks_data has structure: {"timestamp": ..., "hands": [...]}
        if landmarks_data and "hands" in landmarks_data and landmarks_data["hands"]:
            # Use first hand if available (matches training logic)
            hand_features = landmarks_data["hands"][0]

            # Ensure we have exactly 176 features
            if len(hand_features) < self.num_features:
                # Pad with zeros if fewer features
                hand_features = hand_features + [0.0] * (
                    self.num_features - len(hand_features)
                )
            elif len(hand_features) > self.num_features:
                # Truncate if more features
                hand_features = hand_features[: self.num_features]

            self.frame_buffer.append(hand_features)
        else:
            # No hands detected, add zero vector
            self.frame_buffer.append([0.0] * self.num_features)

        # Keep buffer within reasonable size
        if len(self.frame_buffer) > self.max_inference_frames:
            self.frame_buffer.pop(0)

    def can_predict(self):
        """Check if we have enough frames for prediction"""
        return len(self.frame_buffer) >= self.min_inference_frames

    def predict_gesture(self):
        """Make gesture prediction from current frame buffer"""
        if not self.can_predict():
            return None, 0.0

        try:
            # Prepare sequence from buffer
            sequence = np.array(self.frame_buffer, dtype=np.float32)

            # Normalize features using the scaler
            sequence = self.scaler.transform(sequence)

            # Pad or truncate to match training sequence length
            if len(sequence) > self.sequence_length:
                # Take the most recent frames
                sequence = sequence[-self.sequence_length :]
            elif len(sequence) < self.sequence_length:
                # Pad with the last frame
                last_frame = (
                    sequence[-1]
                    if len(sequence) > 0
                    else np.zeros(self.num_features, dtype=np.float32)
                )
                padding_needed = self.sequence_length - len(sequence)
                padding = np.tile(last_frame, (padding_needed, 1))
                sequence = np.vstack([sequence, padding])

            # Reshape for model input: (batch_size, sequence_length, features)
            sequence = sequence.reshape(1, self.sequence_length, self.num_features)

            # Make prediction
            prediction = self.model.predict(sequence, verbose=0)

            # Get predicted class and confidence
            if len(self.label_encoder.classes_) == 2:
                # Binary classification
                confidence = float(prediction[0][0])
                predicted_class = int(confidence > 0.5)
                confidence = confidence if predicted_class == 1 else 1 - confidence
            else:
                # Multi-class classification
                predicted_class = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_class])

            # Convert to gesture name
            gesture_name = self.label_encoder.classes_[predicted_class]

            return gesture_name, confidence

        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, 0.0

    def smart_clear(self, frames_to_pop):
        for _ in range(0, frames_to_pop):
            self.frame_buffer.pop(0)

def main():
    """Main function for real-time gesture recognition"""

    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

    # MediaPipe setup
    mp_hands = mp.solutions.hands
    hands_landmarker = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # Initialize gesture recognizer
    gesture_recognizer = GestureRecognizer(
        model_path=os.path.join(
            TRAINED_MODEL_DIR, "rnn_bidirectional_lstm_landmarks_gesture_model.keras"
        ),
        scaler_path=os.path.join(TRAINED_MODEL_DIR, "scaler.pkl"),
        label_encoder_path=os.path.join(TRAINED_MODEL_DIR, "label_encoder.pkl"),
    )

    frame_count = 0
    last_prediction = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            # Extract landmarks from current frame
            frame_landmarks = extract_3D_landmarks_from_frame(
                frame, hands_landmarker, use_enhanced_features=True
            )

            # Add landmarks to recognizer buffer
            gesture_recognizer.add_frame_landmarks(frame_landmarks)

            frame_count += 1

            # Make prediction if we have enough frames
            if (
                gesture_recognizer.can_predict() and frame_count % 15 == 0
            ): # inference every 15 frames
                prediction, confidence = gesture_recognizer.predict_gesture()

                if prediction and confidence > gesture_recognizer.min_confidence:
                    gesture_recognizer.smart_clear(10)
                    frame_count -= 10
                    
                    
                    if prediction != last_prediction:
                        print(f"Gesture: {prediction} (Confidence: {confidence:.2f})")
                    last_prediction = prediction
                    # Draw prediction on frame
                    cv2.putText(
                        frame,
                        f"{prediction}: {confidence:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

            # Draw hand landmarks on frame for debugging
            if (
                frame_landmarks
                and "hands" in frame_landmarks
                and frame_landmarks["hands"]
            ):
                # Simple visualization - draw a circle where hands are detected
                cv2.putText(
                    frame,
                    f"Hand detected ({len(frame_landmarks['hands'])} hands)",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

            # Display frame
            cv2.imshow("Gesture Recognition", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during inference: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands_landmarker.close()


if __name__ == "__main__":
    main()
