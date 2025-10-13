import whisperx

device = "cpu"
audio_file = "audio.mp3"
batch_size = 16
compute_type = "float32"

model = whisperx.load_model("large-v2", device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)

# Transcribe with VAD parameters

# vad_onset default: 0.5
# Setting this lower will make the model more sensitive to detecting speech

# vad_offset default: 0.363
# Setting this higher will make the model less sensitive to detecting the end of speech

result = model.transcribe(
    audio, 
    batch_size=batch_size,
    vad_onset=0.5,  # Main VAD parameter
    vad_offset=0.363  # Optional: controls end of speech detection
)

print(result["segments"])