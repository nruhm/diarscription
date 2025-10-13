import os
import whisper

model = whisper.load_model("large")
result = model.transcribe(r"C:\Users\nathanjruhmann\diarscription\docs\reference\audio\sample-a\diarscription-audio-sample-a.mp3")
print(result["text"])