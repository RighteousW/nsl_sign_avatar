import os

# Directory paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../models")
)
DEPENDENCY_MODEL_DIR = os.path.join(MODEL_DIR, "dependencies")
TRAINED_MODEL_DIR = os.path.join(MODEL_DIR, "trained_models")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
LANDMARKS_DIR = os.path.join(DATA_DIR, "landmarks")

# Camera and capture settings
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
FRAME_RATE = 30
