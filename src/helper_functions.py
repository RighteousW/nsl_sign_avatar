from datetime import datetime
import glob
import os
from pathlib import Path
import pickle
import sys
import time

import cv2
import numpy as np
import tensorflow as tf


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import FRAME_RATE, VIDEO_DIR


def get_all_video_files():
    """Scan VIDEO_DIR and return all video files organized by word."""
    video_files = {}

    if not os.path.exists(VIDEO_DIR):
        print(f"Video directory {VIDEO_DIR} does not exist.")
        return video_files

    # Look for video files in word subdirectories
    for word_dir in os.listdir(VIDEO_DIR):
        word_path = os.path.join(VIDEO_DIR, word_dir)

        if (
            os.path.isdir(word_path)
            and word_dir != "landmarks_dataset"
            and word_dir != "trained_models"
        ):
            # Find all video files in this word directory
            video_patterns = ["*.avi"]
            word_videos = []

            for pattern in video_patterns:
                word_videos.extend(glob.glob(os.path.join(word_path, pattern)))

            if word_videos:
                video_files[word_dir] = word_videos

    return video_files


def create_directory(directory_path):
    """Create directory structure."""
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


def extract_landmarks(hand_landmarker, frame):
    """Extract 3D hand landmarks from frame - XYZ coordinates."""
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands
    hand_results = hand_landmarker.process(rgb_frame)

    landmarks_data = {"timestamp": time.time(), "hands": []}

    # Extract hand landmarks - XYZ coordinates for better discrimination
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            hand_data = []
            for landmark in hand_landmarks.landmark:
                # Include all three coordinates: X, Y, Z
                hand_data.extend([landmark.x, landmark.y, landmark.z])
            landmarks_data["hands"].append(hand_data)
    return landmarks_data


def extract_enhanced_features(hand_landmarks):
    """Extract additional geometric features from 3D landmarks."""
    if len(hand_landmarks) < 63:  # 21 landmarks * 3 coords
        return hand_landmarks

    # Reshape to get individual landmark coordinates
    coords = []
    for i in range(0, len(hand_landmarks), 3):
        if i + 2 < len(hand_landmarks):
            coords.append(
                [hand_landmarks[i], hand_landmarks[i + 1], hand_landmarks[i + 2]]
            )

    if len(coords) < 21:
        return hand_landmarks

    coords = np.array(coords)
    enhanced_features = list(hand_landmarks)  # Start with original XYZ

    # Add finger tip distances (important for U vs V discrimination)
    fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            tip1 = coords[fingertips[i]]
            tip2 = coords[fingertips[j]]
            distance = np.linalg.norm(tip1 - tip2)
            enhanced_features.append(distance)

    # Add finger angles relative to palm
    palm_center = coords[0]  # Wrist as palm reference
    for tip_idx in fingertips:
        tip_pos = coords[tip_idx]
        # Calculate angle from palm to fingertip
        vector = tip_pos - palm_center
        # Add normalized vector components
        norm = np.linalg.norm(vector)
        if norm > 0:
            normalized_vector = vector / norm
            enhanced_features.extend(normalized_vector)
        else:
            enhanced_features.extend([0.0, 0.0, 0.0])

    return enhanced_features


def extract_3D_landmarks_from_video_file(
    video_path, word, landmarks_dir, hands_landmarker, use_enhanced_features=True
):
    """Process a single video file and extract 3D landmarks with optional enhanced features."""
    # Setup output path
    word_landmarks_dir = os.path.join(landmarks_dir, word)
    os.makedirs(word_landmarks_dir, exist_ok=True)

    # Generate output filename (using .pkl for binary storage)
    video_name = Path(video_path).stem
    suffix = "_3d_enhanced" if use_enhanced_features else "_3d"
    output_path = os.path.join(
        word_landmarks_dir, f"{video_name}_landmarks{suffix}.pkl"
    )

    # Check if already processed
    if os.path.exists(output_path):
        print(f"Landmarks already exist for {video_name}")
        overwrite = input("Overwrite existing landmarks? (y/n): ").lower()
        if overwrite != "y":
            print("Skipping...")
            return False

    print(f"Processing: {os.path.basename(video_path)}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video info: {total_frames} frames at {fps:.2f} FPS")

    landmarks_sequence = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            landmarks_sequence.append(extract_3D_landmarks_from_frame(frame, hands_landmarker, use_enhanced_features))

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        cap.release()
        cv2.destroyAllWindows()
        return False

    finally:
        cap.release()

    # Validate landmarks sequence
    if not landmarks_sequence:
        print("No landmarks extracted from video.")
        return False

    # Calculate feature dimensions
    if use_enhanced_features:
        base_features = 63  # 21 landmarks * 3 coords
        fingertip_distances = 10  # 5 fingertips choose 2
        finger_angles = 15  # 5 fingers * 3 coords each
        total_features_per_hand = (
            base_features + fingertip_distances + finger_angles
        )  # 88
        total_features = total_features_per_hand * 2  # 176 for 2 hands
        feature_type = "3D_Enhanced"
    else:
        total_features = 126  # 2 hands * 21 landmarks * 3 coords
        feature_type = "3D_Basic"

    # Save landmarks dataset using pickle for binary storage
    dataset = {
        "word": word,
        "source_video": os.path.basename(video_path),
        "total_frames": len(landmarks_sequence),
        "original_fps": fps,
        "target_fps": FRAME_RATE,
        "landmarks_sequence": landmarks_sequence,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "coordinate_system": feature_type,
            "hands_per_frame": "max_2",
            "hand_landmarks_per_hand": 21,
            "feature_dimensions": {
                "hands": f"2_hands_x_{total_features_per_hand if use_enhanced_features else 63}_features = {total_features}",
                "total_per_frame": total_features,
                "enhanced_features": use_enhanced_features,
                "note": "XYZ coordinates with optional geometric enhancements for better discrimination",
            },
        },
    }

    try:
        with open(output_path, "wb") as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✅ Saved landmarks: {os.path.basename(output_path)}")
        print(f"   Frames processed: {len(landmarks_sequence)}")
        print(
            f"   Feature dimensions: {dataset['metadata']['feature_dimensions']['hands']}"
        )
        print(f"   Feature type: {feature_type}")
        return True

    except Exception as e:
        print(f"❌ Error saving landmarks: {e}")
        return False


def extract_3D_landmarks_from_frame(frame, hands_landmarker, use_enhanced_features=True):
    landmarks_data = extract_landmarks(hands_landmarker, frame)

    # Optionally enhance features
    if use_enhanced_features and landmarks_data["hands"]:
        enhanced_hands = []
        for hand_data in landmarks_data["hands"]:
            enhanced_hand = extract_enhanced_features(hand_data)
            enhanced_hands.append(enhanced_hand)
        landmarks_data["hands"] = enhanced_hands
    return landmarks_data


def save_model(model, filepath):
    """
    Save model using the modern Keras format (.keras) instead of legacy HDF5 (.h5)
    """
    if filepath.endswith(".h5"):
        filepath = filepath.replace(".h5", ".keras")

    try:
        model.save(filepath)
        print(f"Model saved in: {filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")
        # Fallback to HDF5 if modern format fails
        fallback_path = filepath.replace(".keras", ".h5")
        model.save(fallback_path)
        print(f"Fallback: Model saved in HDF5 format: {fallback_path}")

    return filepath


def load_model(filepath):
    """
    Load model from either .keras or .h5 format
    """
    import os

    # Try .keras first (modern format)
    keras_path = (
        filepath.replace(".h5", ".keras") if filepath.endswith(".h5") else filepath
    )
    if os.path.exists(keras_path):
        try:
            return tf.keras.models.load_model(keras_path)
        except Exception as e:
            print(f"Could not load .keras format: {e}")

    # Fallback to .h5 format
    h5_path = (
        filepath.replace(".keras", ".h5") if filepath.endswith(".keras") else filepath
    )
    if os.path.exists(h5_path):
        try:
            return tf.keras.models.load_model(h5_path)
        except Exception as e:
            print(f"Could not load .h5 format: {e}")

    raise FileNotFoundError(
        f"Could not find model file at {filepath} (tried both .keras and .h5 formats)"
    )
