import whisper
import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf
import os

# Load Whisper model
model = whisper.load_model("base")

# Audio settings
RATE = 16000
DURATION = 3

print("Press Ctrl+C to stop")

try:
    while True:
        # Record audio
        audio = sd.rec(
            int(RATE * DURATION), samplerate=RATE, channels=1, dtype=np.float32
        )
        sd.wait()

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, audio, RATE)

        # Transcribe (disable FP16 to avoid warnings)
        result = model.transcribe(temp_file.name, fp16=False)
        text = result["text"].strip()

        # Clean up temp file
        os.unlink(temp_file.name)

        if text:
            print(text)

except KeyboardInterrupt:
    print("\nStopped.")
