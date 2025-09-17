###############################################################################
# Created 9/16/25
# Nathan Ruhmann
#
# Compact version with tensor size mismatch fixes
# This is version 2.1 of the rough draft
###############################################################################

# Import mods 
from pyannote.audio import Pipeline 
import torch
import whisper
import warnings
from datetime import timedelta
# Imports used for file preprocessing 
import librosa
import soundfile as sf

warnings.filterwarnings("ignore", category=UserWarning)

# Convert seconds to SRT time format (HH:MM:SS,mmm)
def seconds_to_srt_time(seconds):
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(int(td.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((td.total_seconds() % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# Create SRT file with speaker diarization
def create_srt_with_speakers(audio_file, output_file="output.srt"):
    # Preprocess audio to fix tensor size issues
    print("Preprocessing audio...")
    audio, sr = librosa.load(audio_file, sr=16000)
    audio = librosa.util.normalize(audio)
    processed_file = "temp_processed.wav"
    sf.write(processed_file, audio, sr)
    
    try:
        # Load models from HF: speaker-diarization-3.1 
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="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"
        )
        
        if torch.cuda.is_available():
            diarization_pipeline.to(torch.device("cuda"))
        
        whisper_model = whisper.load_model("base") # Can be adjusted but base is fine 
        
        print("Running speaker diarization...")
        try:
            diarization = diarization_pipeline(processed_file)
        except RuntimeError as e:
            if "Sizes of tensors must match" in str(e):
                print("Tensor error - using transcription only...") # Backup if the error persists from previous version 
                return create_transcription_only(processed_file, output_file, whisper_model)
            raise e
        
        print("Running speech transcription...")
        transcription = whisper_model.transcribe(processed_file) # Use whisper to transcribe the processed file 
        
        # Combine diarization and transcription
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True): #
            for segment in transcription['segments']:
                if (segment['start'] < turn.end and segment['end'] > turn.start):
                    segments.append({
                        'start': max(segment['start'], turn.start),   # Use overlapping start time 
                        'end': min(segment['end'], turn.end),         # Use overlapping end time 
                        'text': segment['text'].strip(),              # Transcribed text 
                        'speaker': speaker                            # Assigned speaker 
                    }) # Needs to see duplicate entries 
        
        segments.sort(key=lambda x: x['start']) # Sort segments by start times 
        write_srt(segments, output_file) # Finally write to SRT file 
        
    except Exception as e:
        print(f"Diarization failed: {e}") # Print error if diarization fails 
        print("Falling back to transcription only...") # Falls back to only whisper 
        create_transcription_only(processed_file, output_file)
    
    finally:
        # Clean up
        import os # If the file exists then remove it before it creates the new one 
        if os.path.exists(processed_file):
            os.remove(processed_file)

# Fallback: transcription without speaker identification
def create_transcription_only(audio_file, output_file, whisper_model=None):
    if whisper_model is None:
        whisper_model = whisper.load_model("base")
    
    transcription = whisper_model.transcribe(audio_file)
    segments = [{
        'start': seg['start'],
        'end': seg['end'], 
        'text': seg['text'].strip(),
        'speaker': 'Speaker'
    } for seg in transcription['segments']]
    
    write_srt(segments, output_file) # Write to SRT file, passing the list of segments and the output file name 

# Write segments to SRT file
def write_srt(segments, output_file): 
    with open(output_file, 'w', encoding='utf-8') as f: # Creates the output file 
        for i, segment in enumerate(segments, 1): # For each segment in the list, display the start time, end time, and speaker text 
            start_time = seconds_to_srt_time(segment['start'])         
            end_time = seconds_to_srt_time(segment['end'])
            speaker_text = f"[{segment['speaker']}] {segment['text']}"
            
            f.write(f"{i}\n{start_time} --> {end_time}\n{speaker_text}\n\n") # At the end, write the start time to end time 
    
    print(f"SRT file created: {output_file}") 

create_srt_with_speakers("TeamsMeeting.wav", "meeting_transcript.srt") #Audio file, Transcript name 