import os
import sys
import cv2
import time
from contextlib import contextmanager
from datetime import datetime


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import FRAME_RATE, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_DIR


@contextmanager
def video_capture_context(device_id=0):
    """Context manager for video capture with proper cleanup."""
    capture = cv2.VideoCapture(device_id)
    if not capture.isOpened():
        raise Exception(f"Could not open video device {device_id}")

    try:
        yield capture
    finally:
        capture.release()



class FrameTimer:
    """Handles precise frame timing for consistent frame rates."""

    def __init__(self, target_fps):
        self.target_interval = 1.0 / target_fps
        self.next_frame_time = time.time()

    def wait_for_next_frame(self):
        """Sleep until it's time for the next frame."""
        current_time = time.time()
        sleep_time = self.next_frame_time - current_time

        if sleep_time > 0:
            time.sleep(sleep_time)

        # Schedule next frame (prevents drift)
        self.next_frame_time += self.target_interval


class VideoRecorder:
    """Records videos and manages video file structure."""
    
    def __init__(self, word):
        self.word = word
        self.video_writer = None
        self.video_writer_flipped = None
        self.video_path = None
        self.video_path_flipped = None
        self.frame_count = 0
        self.setup_video_structure()
        
    def setup_video_structure(self):
        """Create directory structure for videos."""
        # Create word-specific directory
        word_dir = os.path.join(VIDEO_DIR, self.word)
        os.makedirs(word_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.word}_{timestamp}.avi"
        filename_flipped = f"{self.word}_{timestamp}_flipped.avi"
        
        self.video_path = os.path.join(word_dir, filename)
        self.video_path_flipped = os.path.join(word_dir, filename_flipped)
    
    def start_recording(self, width, height, fps):
        """Start video recording with specified parameters."""
        # Define the codec and create VideoWriter objects
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        # Original video writer
        self.video_writer = cv2.VideoWriter(
            self.video_path, fourcc, fps, (width, height))
        
        # Flipped video writer
        self.video_writer_flipped = cv2.VideoWriter(
            self.video_path_flipped, fourcc, fps, (width, height))
        
        if not self.video_writer.isOpened():
            raise Exception(f"Could not open video writer for {self.video_path}")
        
        if not self.video_writer_flipped.isOpened():
            raise Exception(f"Could not open video writer for {self.video_path_flipped}")
        
        self.frame_count = 0
        print(f"Started recording:")
        print(f"  Original: {os.path.basename(self.video_path)}")
        print(f"  Flipped:  {os.path.basename(self.video_path_flipped)}")
    
    def add_frame(self, frame):
        """Add a frame to both original and flipped videos."""
        if self.video_writer is not None and self.video_writer_flipped is not None:
            # Write original frame
            self.video_writer.write(frame)
            
            # Write horizontally flipped frame
            flipped_frame = cv2.flip(frame, 1)  # 1 = horizontal flip
            self.video_writer_flipped.write(flipped_frame)
            
            self.frame_count += 1
    
    def stop_recording(self):
        """Stop recording and save both videos."""
        saved_paths = []
        
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            saved_paths.append(self.video_path)
            print(f"Original video saved: {os.path.basename(self.video_path)}")
        
        if self.video_writer_flipped is not None:
            self.video_writer_flipped.release()
            self.video_writer_flipped = None
            saved_paths.append(self.video_path_flipped)
            print(f"Flipped video saved:  {os.path.basename(self.video_path_flipped)}")
        
        if saved_paths:
            print(f"Total frames recorded: {self.frame_count}")
            return saved_paths
        return None
    
    def get_new_recorder(self, word):
        """Create a new recorder instance for the same word."""
        return VideoRecorder(word)


def get_word_from_user():
    """Get a word from the user for video organization."""
    while True:
        word = input("Enter a word for this video recording: ").strip()
        if word:
            # Clean the word to be filesystem-safe
            cleaned_word = "".join(c for c in word if c.isalnum() or c in ('-', '_')).lower()
            if cleaned_word:
                return cleaned_word
            else:
                print("Please enter a valid word (letters, numbers, hyphens, underscores only).")
        else:
            print("Please enter a word.")


def setup_camera(capture, width, height):
    """Configure camera settings and return actual dimensions."""
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Get actual resolution from the camera
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return actual_width, actual_height


def main():
    """Main video recording loop."""
    # Get word from user
    word = get_word_from_user()
    
    print(f"Recording videos for word: '{word}'")
    print("Instructions:")
    print("- Position yourself in frame")
    print("- Press SPACE to start recording")
    print("- Press SPACE again to stop recording")
    print("- Press 'q' to quit")
    print("- Press 's' to save current recording and start new one")
    print("- Each recording creates TWO videos: original and horizontally flipped")
    
    # Initialize components
    frame_timer = FrameTimer(FRAME_RATE)
    video_recorder = VideoRecorder(word)
    
    recording = False
    count = 0
    can_record = True
    
    try:
        with video_capture_context(0) as capture:
            # Setup camera
            width, height = setup_camera(capture, VIDEO_WIDTH, VIDEO_HEIGHT)
            print(f"Camera resolution: {width}x{height}")
            
            while True:
                # Wait for proper timing
                frame_timer.wait_for_next_frame()
                
                # Capture frame
                ret, frame = capture.read()
                if not ret:
                    print("Failed to grab frame.")
                    break
                
                # Create display frame (don't modify the original for recording)
                display_frame = frame.copy()
                
                # Add recording indicator
                status_color = (0, 255, 0) if recording else (0, 0, 255)
                status_text = "RECORDING" if recording else "READY"
                cv2.putText(display_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                
                # Show frame count if recording
                if recording:
                    frame_count_text = f"Frames: {video_recorder.frame_count}"
                    cv2.putText(display_frame, frame_count_text, (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show word being recorded
                word_text = f"Word: {word}"
                cv2.putText(display_frame, word_text, (10, height - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Record frame if recording is active
                if recording:
                    video_recorder.add_frame(frame)
                
                # Display frame
                cv2.imshow(f'Video Recording - {word}', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    # Stop recording if active before quitting
                    if recording:
                        video_recorder.stop_recording()
                    break
                elif key == ord(' '):  # Space to toggle recording
                    if not recording and can_record:
                        # Start recording
                        video_recorder.start_recording(width, height, FRAME_RATE)
                        recording = True
                        print("Started recording...")
                    else:
                        # Stop recording
                        video_recorder.stop_recording()
                        recording = False
                        print("Stopped recording.")
                elif key == ord('s'):  # Save current recording and start new
                    if recording:
                        # Stop current recording
                        saved_paths = video_recorder.stop_recording()
                        recording = False
                        if saved_paths:
                            print(f"Saved {len(saved_paths)} videos:")
                            for path in saved_paths:
                                print(f"  - {os.path.basename(path)}")
                        
                        # Create new recorder for next video
                        video_recorder = video_recorder.get_new_recorder(word)
                        count+=1
                        if count >= 50:
                            cv2.destroyAllWindows()
                            count = 0
                            can_record = False
                            word = get_word_from_user()
                            video_recorder = video_recorder.get_new_recorder(word)
                            can_record = True
                        print("Ready for next recording. Press SPACE to start.")
                    else:
                        print("Not currently recording.")
                        
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        if recording:
            video_recorder.stop_recording()


if __name__ == "__main__":
    main()