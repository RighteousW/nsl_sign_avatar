import os
import pickle
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    GRU,
    Dense,
    Dropout,
    BatchNormalization,
    Bidirectional,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import glob
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import LANDMARKS_DIR, TRAINED_MODEL_DIR, FRAME_RATE


class LandmarkDataGenerator(tf.keras.utils.Sequence):
    """
    Loads and processes landmark files on-demand during training.
    """

    def __init__(
        self,
        file_paths,
        labels,
        num_classes,
        batch_size=32,
        sequence_length=60,
        shuffle=True,
        augment=False,
        scaler=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.file_paths = file_paths
        self.labels = labels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.augment = augment
        self.scaler = scaler
        self.indices = np.arange(len(self.file_paths))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.file_paths))
        batch_indices = self.indices[start_idx:end_idx]

        # Initialize batch arrays with fixed shapes
        batch_size_actual = len(batch_indices)
        X_batch = np.zeros(
            (batch_size_actual, self.sequence_length, 176), dtype=np.float32
        )
        y_batch = np.zeros((batch_size_actual, self.num_classes), dtype=np.float32)

        valid_samples = 0

        for i, idx in enumerate(batch_indices):
            # Load landmark file
            sequence, label = self._load_landmark_file(idx)

            if sequence is not None:
                # Process sequence
                processed_sequence = self._process_sequence(sequence)

                if processed_sequence is not None and processed_sequence.shape == (
                    self.sequence_length,
                    176,
                ):
                    X_batch[valid_samples] = processed_sequence
                    # Convert to one-hot encoding
                    y_batch[valid_samples, self.labels[idx]] = 1.0
                    valid_samples += 1
                else:
                    print(
                        f"Warning: Invalid sequence shape from file {self.file_paths[idx]}"
                    )

        # Return only valid samples
        if valid_samples == 0:
            print(f"Warning: No valid samples in batch {index}")
            # Return minimal batch to avoid training failure
            X_batch = np.zeros((1, self.sequence_length, 176), dtype=np.float32)
            y_batch = np.zeros((1, self.num_classes), dtype=np.float32)
            y_batch[0, 0] = 1.0
            return X_batch, y_batch

        return X_batch[:valid_samples], y_batch[:valid_samples]

    def _load_landmark_file(self, idx):
        """Load a single landmark file"""
        try:
            file_path = self.file_paths[idx]

            with open(file_path, "rb") as f:
                data = pickle.load(f)

            landmarks_sequence = data["landmarks_sequence"]

            # Extract landmark features from each frame
            sequence_features = []

            for frame_data in landmarks_sequence:
                if frame_data and "hands" in frame_data and frame_data["hands"]:
                    # Use first hand if available
                    hand_features = frame_data["hands"][0]

                    # Ensure we have exactly 176 features
                    if len(hand_features) < 176:
                        # Pad with zeros if fewer features
                        hand_features = hand_features + [0.0] * (
                            176 - len(hand_features)
                        )
                    elif len(hand_features) > 176:
                        # Truncate if more features
                        hand_features = hand_features[:176]

                    sequence_features.append(hand_features)
                else:
                    # No hands detected, use zero vector
                    sequence_features.append([0.0] * 176)

            if len(sequence_features) == 0:
                return None, None

            return np.array(sequence_features, dtype=np.float32), data["word"]

        except Exception as e:
            print(f"Error loading {self.file_paths[idx]}: {e}")
            return None, None

    def _process_sequence(self, sequence):
        """Process sequence: normalize, pad/truncate, augment"""
        if sequence is None or len(sequence) == 0:
            return None

        try:
            # Ensure sequence is the right shape
            if sequence.shape[1] != 176:
                print(
                    f"Warning: Sequence has {sequence.shape[1]} features, expected 176"
                )
                return None

            # Normalize features if scaler is provided
            if self.scaler is not None:
                sequence = self.scaler.transform(sequence)

            # Pad or truncate to fixed length
            if len(sequence) > self.sequence_length:
                # Truncate: take evenly spaced frames
                indices = np.linspace(
                    0, len(sequence) - 1, self.sequence_length, dtype=int
                )
                sequence = sequence[indices]
            elif len(sequence) < self.sequence_length:
                # Pad with the last frame
                last_frame = (
                    sequence[-1]
                    if len(sequence) > 0
                    else np.zeros(176, dtype=np.float32)
                )
                padding_needed = self.sequence_length - len(sequence)
                padding = np.tile(last_frame, (padding_needed, 1))
                sequence = np.vstack([sequence, padding])

            # Apply augmentation if enabled
            if self.augment:
                sequence = self._augment_sequence(sequence)

            # Ensure final shape is correct
            if sequence.shape != (self.sequence_length, 176):
                print(
                    f"Warning: Final sequence shape {sequence.shape}, expected ({self.sequence_length}, 176)"
                )
                return None

            return sequence.astype(np.float32)

        except Exception as e:
            print(f"Error processing sequence: {e}")
            return None

    def _augment_sequence(self, sequence):
        """Apply data augmentation to sequence"""
        try:
            # Add small amount of noise
            noise = np.random.normal(0, 0.001, sequence.shape).astype(np.float32)
            sequence = sequence + noise

            # Random time scaling (speed up or slow down slightly)
            if np.random.random() < 0.3:
                scale_factor = np.random.uniform(0.9, 1.1)
                new_length = max(1, int(len(sequence) * scale_factor))

                if new_length != len(sequence):
                    indices = np.linspace(0, len(sequence) - 1, new_length)
                    sequence = np.array([sequence[int(i)] for i in indices])

                    # Re-pad/truncate after scaling
                    if len(sequence) > self.sequence_length:
                        indices = np.linspace(
                            0, len(sequence) - 1, self.sequence_length, dtype=int
                        )
                        sequence = sequence[indices]
                    elif len(sequence) < self.sequence_length:
                        last_frame = (
                            sequence[-1]
                            if len(sequence) > 0
                            else np.zeros(176, dtype=np.float32)
                        )
                        padding_needed = self.sequence_length - len(sequence)
                        padding = np.tile(last_frame, (padding_needed, 1))
                        sequence = np.vstack([sequence, padding])

            return sequence.astype(np.float32)

        except Exception as e:
            print(f"Error in augmentation: {e}")
            return sequence

    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


def scan_landmark_files(landmarks_dir):
    """
    Scan the landmarks directory and return file paths with their corresponding words.
    """
    print(f"Scanning landmark files in: {landmarks_dir}")

    if not os.path.exists(landmarks_dir):
        print(f"Error: Landmarks directory {landmarks_dir} does not exist!")
        return [], [], {}

    file_paths = []
    words = []
    word_counts = {}

    # Scan word directories
    for word_dir in os.listdir(landmarks_dir):
        word_path = os.path.join(landmarks_dir, word_dir)

        if not os.path.isdir(word_path):
            continue

        print(f"Scanning word: {word_dir}")

        # Find landmark files (both basic and enhanced)
        landmark_files = glob.glob(os.path.join(word_path, "*_landmarks_*.pkl"))

        # Prefer enhanced features if available
        enhanced_files = [f for f in landmark_files if "enhanced" in f]
        basic_files = [f for f in landmark_files if "enhanced" not in f]

        files_to_use = enhanced_files if enhanced_files else basic_files

        for file_path in files_to_use:
            file_paths.append(file_path)
            words.append(word_dir)

        word_counts[word_dir] = len(files_to_use)
        print(f"  Found {len(files_to_use)} landmark files")

    print(f"\nTotal files found: {len(file_paths)}")
    print(f"Total words: {len(word_counts)}")
    print("Word distribution:", word_counts)

    return file_paths, words, word_counts


def create_feature_scaler(file_paths, sample_size=1000):
    """
    Create a StandardScaler by sampling from the landmark files.
    """
    print("Creating feature scaler...")

    sample_features = []
    files_to_sample = min(len(file_paths), sample_size // 30)

    sampled_files = np.random.choice(file_paths, files_to_sample, replace=False)

    for file_path in sampled_files:
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            # Sample a few frames from this file
            landmarks_sequence = data["landmarks_sequence"]
            sample_frames = min(10, len(landmarks_sequence))

            for frame_data in landmarks_sequence[:sample_frames]:
                if frame_data and "hands" in frame_data and frame_data["hands"]:
                    hand_features = frame_data["hands"][0]

                    # Ensure consistent feature length
                    if len(hand_features) < 176:
                        hand_features = hand_features + [0.0] * (
                            176 - len(hand_features)
                        )
                    elif len(hand_features) > 176:
                        hand_features = hand_features[:176]

                    sample_features.append(hand_features)
                else:
                    # Add zero vector for missing hands
                    sample_features.append([0.0] * 176)
        except Exception as e:
            print(f"Error sampling from {file_path}: {e}")
            continue

    if len(sample_features) == 0:
        print("Warning: No features sampled. Using identity scaler.")
        return None

    print(f"Sampled {len(sample_features)} frames for scaler fitting")

    scaler = StandardScaler()
    scaler.fit(sample_features)

    print("Feature scaler created successfully")
    return scaler


def build_rnn_model(input_shape, num_classes, model_type="lstm"):
    """
    Build RNN model for gesture recognition with pruning-friendly architecture.
    """
    print(f"Building {model_type.upper()} model...")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")

    model = Sequential()

    if model_type == "lstm":
        model.add(
            LSTM(128, return_sequences=True, input_shape=input_shape, name="lstm_1")
        )
        model.add(Dropout(0.3, name="dropout_1"))
        model.add(BatchNormalization(name="batch_norm_1"))

        model.add(LSTM(64, return_sequences=False, name="lstm_2"))
        model.add(Dropout(0.3, name="dropout_2"))
        model.add(BatchNormalization(name="batch_norm_2"))

    elif model_type == "gru":
        model.add(
            GRU(128, return_sequences=True, input_shape=input_shape, name="gru_1")
        )
        model.add(Dropout(0.3, name="dropout_1"))
        model.add(BatchNormalization(name="batch_norm_1"))

        model.add(GRU(64, return_sequences=False, name="gru_2"))
        model.add(Dropout(0.3, name="dropout_2"))
        model.add(BatchNormalization(name="batch_norm_2"))

    elif model_type == "bidirectional_lstm":
        model.add(
            Bidirectional(
                LSTM(64, return_sequences=True),
                input_shape=input_shape,
                name="bi_lstm_1",
            )
        )
        model.add(Dropout(0.3, name="dropout_1"))
        model.add(BatchNormalization(name="batch_norm_1"))

        model.add(Bidirectional(LSTM(32, return_sequences=False), name="bi_lstm_2"))
        model.add(Dropout(0.3, name="dropout_2"))
        model.add(BatchNormalization(name="batch_norm_2"))

    # Dense layers with names for better pruning control
    model.add(Dense(64, activation="relu", name="dense_1"))
    model.add(Dropout(0.4, name="dropout_3"))
    model.add(Dense(32, activation="relu", name="dense_2"))
    model.add(Dropout(0.3, name="dropout_4"))

    # Output layer
    if num_classes == 2:
        model.add(Dense(1, activation="sigmoid", name="output"))
        loss = "binary_crossentropy"
    else:
        model.add(Dense(num_classes, activation="softmax", name="output"))
        loss = "categorical_crossentropy"

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=["accuracy"])

    print("Model architecture:")
    model.summary()

    return model


def train_rnn_model():
    # Scan landmark files
    file_paths, words, word_counts = scan_landmark_files(LANDMARKS_DIR)

    if len(file_paths) == 0:
        print("No landmark files found! Please run the landmark processor first.")
        return

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(words)
    num_classes = len(label_encoder.classes_)

    print(f"Classes found: {list(label_encoder.classes_)}")

    # Split data
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths,
        encoded_labels,
        test_size=0.2,
        random_state=42,
        stratify=encoded_labels,
    )

    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")

    # Create feature scaler
    scaler = create_feature_scaler(train_files)

    # Training parameters
    SEQUENCE_LENGTH = int(2 * FRAME_RATE)
    BATCH_SIZE = 32
    EPOCHS = 80

    print(f"Sequence length: {SEQUENCE_LENGTH} frames")
    print(f"Batch size: {BATCH_SIZE}")

    # Create data generators
    train_generator = LandmarkDataGenerator(
        train_files,
        train_labels,
        num_classes,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        shuffle=True,
        augment=True,
        scaler=scaler,
    )

    val_generator = LandmarkDataGenerator(
        val_files,
        val_labels,
        num_classes,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        shuffle=False,
        augment=False,
        scaler=scaler,
    )

    # Build model
    input_shape = (SEQUENCE_LENGTH, 176)

    # Select model type
    model_types = ["lstm", "gru", "bidirectional_lstm"]
    print("\nAvailable RNN model types:")
    for i, model_type in enumerate(model_types, 1):
        print(f"{i}. {model_type.upper()}")

    while True:
        try:
            choice = int(input("Select model type (1-3): ")) - 1
            if 0 <= choice < len(model_types):
                selected_model_type = model_types[choice]
                break
            else:
                print("Invalid choice. Please select 1-3.")
        except ValueError:
            print("Please enter a valid number.")

    model = build_rnn_model(input_shape, num_classes, selected_model_type)

    # Setup callbacks for initial training
    os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)

    model_name = f"rnn_{selected_model_type}_landmarks_gesture_model"
    model_path = os.path.join(TRAINED_MODEL_DIR, f"{model_name}.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            model_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
    ]

    # Initial Training
    print(f"\nInitial Training ({EPOCHS} epochs)...")
    print(f"Model will be saved as: {model_name}.keras")

    start_time = time.time()

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    initial_training_time = time.time() - start_time

    # Evaluate initial model
    print("\n📊 Evaluating initial model...")
    _, train_acc = model.evaluate(train_generator, verbose=0)
    _, val_acc = model.evaluate(val_generator, verbose=0)

    print(f"Initial model - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    model_for_conversion = model
    combined_history = history.history
    pruning_time = 0
    pruned_train_acc = train_acc
    pruned_val_acc = val_acc
    pruning_applied_successfully = False

    # Save comprehensive training artifacts
    training_info = {
        "model_type": selected_model_type,
        "sequence_length": SEQUENCE_LENGTH,
        "num_features": 176,
        "num_classes": num_classes,
        "label_encoder_classes": label_encoder.classes_.tolist(),
        "training_files": len(train_files),
        "validation_files": len(val_files),
        "initial_training_time_minutes": initial_training_time / 60,
        "pruning_time_minutes": pruning_time / 60,
        "initial_train_accuracy": train_acc,
        "initial_val_accuracy": val_acc,
        "final_train_accuracy": pruned_train_acc,
        "final_val_accuracy": pruned_val_acc,
        "word_counts": word_counts,
        "pruning_applied": pruning_applied_successfully,
    }

    # After training
    with open(os.path.join(TRAINED_MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(TRAINED_MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    # Plot training history
    plot_comprehensive_training_history(combined_history, selected_model_type)

    return model_for_conversion, combined_history, training_info


def plot_comprehensive_training_history(history, model_type):
    """Plot comprehensive training history including pruning phase"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    suptitle = f"{model_type.upper()} Model Training"

    fig.suptitle(suptitle, fontsize=16, fontweight="bold")

    epochs = range(1, len(history["accuracy"]) + 1)

    # Plot accuracy
    ax1.plot(epochs, history["accuracy"], "b-", label="Training Accuracy", linewidth=2)
    ax1.plot(
        epochs, history["val_accuracy"], "r-", label="Validation Accuracy", linewidth=2
    )
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot loss
    ax2.plot(epochs, history["loss"], "b-", label="Training Loss", linewidth=2)
    ax2.plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_name = f"rnn_{model_type}_comprehensive_training_history"
    plot_path = os.path.join(TRAINED_MODEL_DIR, f"{plot_name}.png")

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nTraining history plot saved: {plot_path}")
    plt.show()


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

    # Run comprehensive training with optimization
    train_rnn_model()
