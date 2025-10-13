import os
import whisper
import time 

audio_file = (r"C:\Users\nathanjruhmann\diarscription\docs\reference\audio\sample-a\diarscription-audio-sample-a.mp3")
models = ["base", "tiny", "small", "medium", "large"]
times = []

for model_name in models:
    model = whisper.load_model(model_name)
    start_time = time.time()
    result = model.transcribe(audio_file)
    elapsed_time = time.time() - start_time
    times.append(elapsed_time)

for totaltime, model_name in zip(times, models):
    print(f"Model {model_name} took {totaltime} seconds to transcribe the audio.")