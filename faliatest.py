import pyaudio
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import numpy as np

# Load the Wav2Vec2 model and processor
model_name = "Falia/wav2vec2-xlsr-300m-voxmg"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Ensure the model uses Float data type
model.to(torch.float32)

# Initialize PyAudio to capture audio from the microphone
p = pyaudio.PyAudio()

# Define audio stream parameters
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate (adjust to match your model's sample rate)
CHUNK = 1024  # Number of frames per buffer
RECORD_SECONDS = 15  # Duration to capture audio

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening...")

audio_input = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    audio_input.append(np.frombuffer(data, dtype=np.int16))

print("Recording finished")

# Close the audio stream
stream.stop_stream()
stream.close()

# Terminate PyAudio
p.terminate()

# Convert audio_input to a single NumPy array
audio_input = np.concatenate(audio_input, axis=0)

# Convert audio_input to Float data type
audio_input = audio_input.astype(np.float32)

# Use the processor to preprocess the audio
input_values = processor(audio_input, return_tensors="pt").input_values

with torch.no_grad():
    logits = model(input_values).logits

# Perform CTC decoding to get the transcription
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print("Transcription:", transcription)
