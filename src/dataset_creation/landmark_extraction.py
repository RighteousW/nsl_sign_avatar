import os
import sys
import time
import mediapipe as mp


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import LANDMARKS_DIR, VIDEO_DIR
from helper_functions import (
    create_directory,
    get_all_video_files,
    extract_3D_landmarks_from_video_file,
)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands_landmarker = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)


def main():
    """Main processing loop."""
    print("NSL Video to 3D Landmarks Dataset Processor")
    print("Enhanced Features: Using XYZ coordinates + geometric features")
    print("Benefits: Better discrimination between similar signs (U vs V)")
    print("Feature Options:")
    print("   - Basic 3D: 126 features (2 hands x 21 landmarks x 3 coords)")
    print("   - Enhanced 3D: 176 features (+ fingertip distances + finger angles)")

    # Setup
    landmarks_dir = create_directory(LANDMARKS_DIR)

    # Get all video files
    print("\n🔍 Scanning for video files...")
    video_files = get_all_video_files()

    if not video_files:
        print("No video files found in the video directory structure.")
        print(f"Expected structure: {VIDEO_DIR}/word_name/*.avi")
        print("Please run the video recorder first to create videos.")
        return

    # Feature extraction options
    print("\nFeature extraction options:")
    print("1. Basic 3D (XYZ coordinates only - 126 features)")
    print("2. Enhanced 3D (XYZ + geometric features - 176 features)")

    while True:
        feature_choice = input("\nEnter choice (1-2): ").strip()
        if feature_choice in ["1", "2"]:
            break
        print("Invalid choice. Please enter 1 or 2.")

    use_enhanced_features = feature_choice == "2"

    # Processing options
    print("\nProcessing options:")
    print("1. Process all videos")
    print("2. Process specific word")

    while True:
        choice = input("\nEnter choice (1-2): ").strip()
        if choice in ["1", "2"]:
            break
        print("Invalid choice. Please enter 1, 2.")

    process_word = None

    if choice == "2":
        print("\nAvailable words:")
        word_list = list(video_files.keys())
        for i, word in enumerate(word_list, 1):
            print(f"  {i}. {word}")

        while True:
            try:
                word_choice = int(input("Enter word number: ")) - 1
                if 0 <= word_choice < len(word_list):
                    process_word = word_list[word_choice]
                    break
                else:
                    print("Invalid number. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    # Calculate processing scope
    if process_word:
        videos_to_process = {process_word: video_files[process_word]}
        total_to_process = len(video_files[process_word])
    else:
        videos_to_process = video_files
        total_to_process = sum(len(videos) for videos in video_files.values())

    feature_info = (
        "Enhanced 3D (176 features)"
        if use_enhanced_features
        else "Basic 3D (126 features)"
    )
    print(f"\nStarting processing of {total_to_process} videos...")
    print(f"Extracting: {feature_info}")

    # Process videos
    processed_count = 0
    skipped_count = 0
    error_count = 0
    start_time = time.time()

    try:
        for word, videos in videos_to_process.items():
            print(f"\nProcessing word: '{word}' ({len(videos)} videos)")

            for i, video_path in enumerate(videos, 1):
                print(f"\n[{i}/{len(videos)}] ", end="")

                try:
                    if extract_3D_landmarks_from_video_file(
                        video_path,
                        word,
                        landmarks_dir,
                        hands_landmarker,
                        use_enhanced_features,
                    ):
                        processed_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    print(f"Error processing {os.path.basename(video_path)}: {e}")
                    error_count += 1

                # Update overall progress
                current_total = processed_count + skipped_count + error_count
                overall_progress = (current_total / total_to_process) * 100
                elapsed_time = time.time() - start_time

                if current_total > 0:
                    eta = (elapsed_time / current_total) * (
                        total_to_process - current_total
                    )
                    print(
                        f"Overall progress: {overall_progress:.1f}% - ETA: {eta/60:.1f} min"
                    )

    except KeyboardInterrupt:
        print("Processing interrupted by user.")

    finally:
        hands_landmarker.close()

    # Final summary
    elapsed_time = time.time() - start_time
    print("Processing complete!")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Videos processed: {processed_count}")
    print(f"Videos skipped: {skipped_count}")
    print(f"Videos with errors: {error_count}")
    print(f"Landmarks dataset saved to: {landmarks_dir}")

    if use_enhanced_features:
        print(f"Enhanced feature dimensions: 176 per frame")
        print(f"   • Base 3D coordinates: 126 features")
        print(f"   • Fingertip distances: 10 features")
        print(f"   • Finger orientation vectors: 15 features")
        print(f"   • Total per hand: 88 features x 2 hands = 176")
    else:
        print(f"Basic 3D feature dimensions: 126 per frame")
        print(f"   • XYZ coordinates: 2 hands x 21 landmarks x 3 coords = 126")

    if processed_count > 0:
        avg_time = elapsed_time / processed_count
        print(f"Average processing time: {avg_time:.1f} seconds per video")
        print(f"Ready for enhanced sign recognition training!")


if __name__ == "__main__":
    main()
