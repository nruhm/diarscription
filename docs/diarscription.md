# Speaker Diarization in Python

Speaker diarization is the process of identifying who speaks when in an audio recording. Python offers several libraries to perform this task effectively. Alongside whisper, there are three common approaches:

## 1. Using pyannote.audio (Version 3.1 current)

pyannote.audio is a powerful open-source toolkit for diarization based on PyTorch. It provides pre-trained models and pipelines for high performance.

### Example of pyannote in action:

```python
from pyannote.audio import Pipeline

# Load pre-trained speaker diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Apply the pipeline to an audio file
diarization = pipeline("path_to_audio_file.wav")

# Print diarization results
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"Speaker {speaker} spoke from {turn.start:.1f}s to {turn.end:.1f}s")
```
## 2. Using pyAudioAnalysis 

pyAudioAnalysis is another library that supports speaker diarization, though it requires more manual setup. 

### Example of pyAudioAnalysis in action:

```python 
from pyAudioAnalysis import audioSegmentation as aS

# Perform speaker diarization 
segments, speakers = aS.speakerDiarization("path_to_audio_file.wav", num_of_speakers=2)

# Print speaker labels for each segment
print(segments)
print(segments)
```

## 3. Using Hugging Face's pyannote Models 

Hugging Face hosts pre-trained models for speaker diarization, which can be fine-tuned for specific datasets. This will require setup on the Hugging Face website, as well as creating a token. 

```python
from pyannote.audio import Pipeline

# Load the Hugging Face model
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Process an audio file
diarization = pipeline("path_to_audio_file.wav")

# Output results
for segment in diarization.itersegments():
    print(segment)
```

### Before taking notes, I tried to create my own program with what I learned through the textbook and whisper, etc.
### Two errors I got while running the rough draft:
- Error processing audio file: Sizes of tensors must match except in dimension 0. Expected size 160000 but got size 157448 for tensor number 27 in the list.
- No HuggingFace token provided 


# The official pyannote github has the following instructions for installation/setup:
1. Install pyannote.audo with pip install pyannote.audio 
2. Accept the https://huggingface.co/pyannote/segmentation-3.0 user conditions
3. Accept the https://huggingface.co/pyannote/speaker-diarization-3.1 user conditions 
4. Create access token at https://huggingface.co/settings/tokens with the following settings:
    - Token type: Finegrained
    - Repositories
        - Check "Read access to contents of all public gated repos you can access" 

# Python example after initial setup: 
```python
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE")

# send pipeline to GPU (when available)
import torch
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline
diarization = pipeline("audio.wav")

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

# Expected output: 
# start=0.2s stop=1.5s speaker_0
# start=1.8s stop=3.9s speaker_1
# start=4.2s stop=5.7s speaker_0
# ...
```

## Advanced pyannote.audio Features 
Pyannote uses a multi-satge pipeline approach:
1. Voice Activity Detection (VAD): Identifies speech vs silence
2. Speaker Change Detection: Finds boundaries between speakers 
3. Speaker Embedding: Creates voice "fingerprints" for each segment
4. Clustering: Groups similar voice embeddings as the same speaker 

## Common Technical Issues in Speaker Diarization
- Overlapping speech
- Short utterances
- Similar voices 
- Background noise
- Variable audio quality 

# Common Errors 

## 1. "Error processing audio file: Sizes of tensors must match except in dimension 0. Expected size 160000 but got size 157448 for tensor number 27 in the list."
The pyannote speaker diarization model expects audio files to have specific length requirements or padding that results in tensors of *exactly* 160,000 elements. 
The rought draft of the diarization project came across this error because the TeamsMeeting.wav file is producing tensors of 157,448 elements, which is about 9.84 seconds of audio, when
pyannote wants 160,000 elements/10 seconds at 16kHz sample rate. 

The solution in the rough draft is using the module librosa, which was not discussed earlier in this research document. The code does the following:
```python
audio, sr = librosa.load(audio_file, sr=16000)
audio = librosa.util.normalize(audio)
processed_file = "temp_processed.wav"
sf.write(processed_file, audio, sr)
```
### librosa.load(audio_file, sr=16000):
This line loads the audio and forces resampling to exactly 16kHz, which is the required sample rate for pyannote. It also strips away any metadata, headers, or encoding quirks from the original audio file. 

### librosa.util.normalize(audio):
This line normalizes the amplitude so the loudest peak reaches exactly 1.0, which ensures consistent volume levels across the entire audio. You can hear a distinct difference if you compare the original audio file to the new formatted one.

### sf.write(processed_file, audio, sr):
This line saves the file as a standard WAV file, regardless of what it was before (mp3, wav, etc.). The new audio file has an exact 16kHz sample rate and has no extraneous metadata. 

After that is ran, processed_file is the new audio file that is used throughout the rest of the program. This is the current solution to the 16kHz error. 

## 2. Repeated lines in the SRT file 
Another common problem I have come across after running the rough draft is that there will be repeated segments. Below is an example from the TeamsMeeting.wav transcription file: 

```
13
00:00:32,700 --> 00:00:32,719
[SPEAKER_01] But yeah, just

14
00:00:32,719 --> 00:00:38,269
[SPEAKER_00] Scrappers and could have been in the way well it floated over the push bar floated over and pulled the other sheat on top of it

15
00:00:32,719 --> 00:00:33,274
[SPEAKER_01] Scrappers and could have been in the way well it floated over the push bar floated over and pulled the other sheat on top of it

16
00:00:38,404 --> 00:00:40,799
[SPEAKER_00] Scrappers and could have been in the way well it floated over the push bar floated over and pulled the other sheat on top of it

17
00:00:42,252 --> 00:00:43,960
[SPEAKER_00] So there was two sheets
```
Since this is just a rough draft, the following code can be accounted for causing this error and can be fixed in a later version: 

```python
for turn, _, speaker in diarization.itertracks(yield_label=True):
    for segment in transcription['segments']:
        if (segment['start'] < turn.end and segment['end'] > turn.start):
            segments.append({...})
```

A single Whisper transcription segment can overlap with multiple diarization "turns", creating duplicate matches. The code also doesn't check if the same text was already added. 
This could also cause problems if fixed because if someone repeats something, it may flag it as a "repeat" and not print it in the transcription. When pyannote is uncertain about speaker changes, it creates overlapping speaker segments. 

### Solutions to fix this problem:
- Remove segments with identical or very similar text by filtering each result comparing it to the one before/after 
- Combine segments that have significant time overlaps 
- When duplicates exist, keep the one with the longer duration by comparing the times
- Prevent segments that are too close in time from being duplicated 

These solutions might also create additional problems as listed before, so more work may have to be put into it.

## 3. Incorrect Speaker Identification 
In the test audio of roughly 2 minutes, there are three distinct speakers. The issue I have come across is that Speaker 0 (the presenter) is the main person speaking in the audio file, with Speaker 1 and 2 making small comments that add up to only a few seconds. This throws off the current version because sometimes the comments are mid-sentence comments that are inserted in whatever Speaker 0 is saying.

This might be fixed if the file size is larger or if multiple people are talking clearly in the audio file, with minimal white noise/interruptions. The current audio file that's being tested has white noise and a lack of multiple participants making multiple comments, enough for pyannote to count them as a seperate speaker. There are times where only a word are two are said, usually as interruptions, and if those words were to be flagged they would just be inserted in the main speaker's dialouge.

Example: The following should have made the third line be Speaker 1 saying "Oh to grab situation" then Speaker 0 corrected him saying "Well, it's a push bar". This should have been broken up. 

```
1
00:00:00,756 --> 00:00:05,860
[SPEAKER_00] Okay, well there was something that caused it to not go down far enough

2
00:00:07,080 --> 00:00:09,599
[SPEAKER_00] Actually, there's some kind of

3
00:00:09,599 --> 00:00:12,519
[SPEAKER_00] Oh to grab situation, well, it's a push bar
```

The following should have made the third line be Speaker 1 saying "Oh to grab situation" then Speaker 0 corrected him saying "Well, it's a push bar". This should have been broken up. 
The only solution I can think of right now is using a different pyannote version (2.1) or researching different models that are specifically trained for meeting scenarios. 

## To do:
- Test multiple files
- Adjust rough draft to find potential solutions 