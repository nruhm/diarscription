Nathan Ruhmann
9/22/25

# USE first audio file (Test1.wav)
# Import mods
import os
import subprocess
from pyannote.audio import Pipeline 
import json
import sys
import whisper
import whisperx
import torch
import scipy.io.wavfile
import warnings
import numpy as np
warnings.filterwarnings("ignore") # Add more if pytorch spam with depreciation warnings 

os.chdir(r'C:\Users\nathanjruhmann\Scripts\audio')

# Transcription: Using whisper transcribe the audio file and save it to a variable 
FUNCTION whisper
    whisper_model = whisper.load_model("base")
    transcription_result = whisper_model.transcribe("AUDIOFILE")
    full_text = transcription_result["text"]

    with open("audio.srt", "w") as srt:
        diarization.write_rttm(srt)
END FUNCTION 

# Number of Speakers: Using pyannote find the amount of speakers and save it to a variable "speakers"
FUNCTION pyannote
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="HF TOKEN")
    diarization = diarization_pipeline("AUDIOFILE")
    speakers = list(diarization.labels())
    speaker_count = len(speakers)

    IF 
        speaker_count < 3
    THEN 
        print(f"Only {speaker_count} speakers detected. Expected: 3")
END FUNCTION

# When speakers speak: Create a dictionary value of every single time stamp and the speaker, text, etc. 
FUNCTION create_dictionary 
    whisper_segments = transcription_result["segments"]

    speaker_dict = {}
    segment_counter = 1

    FOR each segment IN whisper_segments:
        start_time = segment["start"]       
        end_time = segement["end"]
        text = segment["text"]
        segment_midpoint = (start_time + end_time) / 2

        assigned_speaker = "Unknown"                                # Initially. This is before pyannote replaces it

        FOR each turn, _, speaker IN diarization.itertracks(yield_label=True):
            IF turn.start <= segment_midpoint <= turn.end:          # If the diarization information matches whisper's information 
                THEN assigned_speaker = speaker                     # Assigns that speaker to the dictionary 
        
        speaker_dict[segment_counter] = {
            "start_time": start_time,
            "end_time": end_time,
            "speaker": assigned_speaker,
            "text": text
        }

        segment_counter = segment_counter + 1                       # Iterate through every segment in the whisper's transcript
    RETURN speaker_dict
END FUNCTION

# Speaker overlap ranged when multiple people talk: Get the JSON data and find out what volume each speaker averages at 

?? - Export the JSON data from pyannote 

# Separate speakers: Make a dictionary and add each speaker found using pyannote into it 

This was done in a previous step

# Match speakers inrange, match number of speakers detected: Put the two variables against each other and find out if they both equal the same amount of speakers 

This was also done before, but another verification step will be made in the whisperx section. 

# Create audio clips for each speaker: Use pyannote.ami to create X amount of audio clips for each speaker
FUNCTION pyannoteami 
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speech-separation-ami-1.0", use_auth_token="HF TOKEN")

    diarization, sources = pipeline("AUDIOFILE")

    with open("audio.srt", "w") as srt:
        diarization.write_rttm(srt)

    created_audio_files = []

    FOR s, speaker IN enumerate(diarization.labels()):
        filename = f'{speaker}.wav'
        
        audio_data = sources.data[:, s]
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        scipy.io.wavfile.write(filename, 16000, audio_int16)   
        
        created_audio_files.append(filename)                    # add file name to list 
    
    RETURN created_audio_files                                  # We can now use this for whisperx 
END FUNCTION

# Run whisperx through those audio files: Use whisperx to gather the JSON data and store them in an array
FUNCTION run_whisperx_on_files
    
    whisperx_results = []
    FOR each audio_file IN created_audio_files:
        
        command = [
            "whisperx",
            "--model", "large-v2", 
            "--device", "cpu",
            "--compute_type", "float32",
            "--chunk_size", "6",
            "--language", "en",
            "--diarize",
            "--min_speakers", "1",
            "--max_speakers", "1",          # individual speaker files via pyannote.ami 
            "--hf_token", "HF_TOKEN",
            audio_file
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        whisperx_results.append({
            "file": audio_file,
            "command_output": result.stdout,
            "errors": result.stderr
        })
    
    RETURN whisperx_results
END FUNCTION

# Get whisperx data into an array (token, start, end, speaker #): Read the data from before into an array and track those four variables 
FUNCTION create_token_array
    with open("whisperx_output.json", "r") as json_file:
        whisperx_data = json.load(json_file)
    
    token_array = []
    token_counter = 1
    
    # Extract word-level data from word_segments
    FOR each word IN whisperx_data["word_segments"]:
        token_entry = [
            token_counter,          # Unique token number
            word["start"],          # Start time
            word["end"],            # End time
            word["speaker"]         # Speaker ID (SPEAKER_00, SPEAKER_01, SPEAKER_02.)
        ]
        
        token_array.append(token_entry)
        token_counter = token_counter + 1
    
    with open("token_array.json", "w") as output_file:
        json.dump(token_array, output_file, indent=2)
    
    RETURN token_array
END FUNCTION

# Make a txt/srt file with all the data: Create a file with accurate speaker diarazation, combining everything from the steps before. 
# You can also verify this information versus the whisperx srt file 
FUNCTION create_final_srt_file
    import json
    
    with open("token_array.json", "r") as token_file:
        token_array = json.load(token_file)
    
    with open("whisperx_output.json", "r") as whisperx_file: 
        whisperx_data = json.load(whisperx_file)
    
    sorted_tokens = sorted(token_array, key=lambda x: x[1]) # Sorts by start time 
    
    with open("final_transcript.srt", "w", encoding="utf-8") as srt_file:
        FOR i, token IN enumerate(sorted_tokens, 1):
            token_num, start_time, end_time, speaker = token
            word_text = whisperx_data["word_segments"][token_num-1]["word"]
            
            start_srt = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d},{int((start_time%1)*1000):03d}"
            end_srt = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d},{int((end_time%1)*1000):03d}"
            
            srt_file.write(f"{i}\n")
            srt_file.write(f"{start_srt} --> {end_srt}\n")
            srt_file.write(f"[{speaker}] {word_text}\n\n")
END FUNCTION