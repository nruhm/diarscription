import json
import re

def format_time_mm_ss(total_seconds):
    """
    Convert total seconds to MM.SS format where SS is actual seconds (0-59).
    Example: 65.5 seconds = 1 minute 5.5 seconds = 1.05 (not 1.655)
    """
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return minutes + seconds / 100.0

def parse_srt_speakers(srt_text):
    """
    Parse SRT-style speaker data and extract speaker segments with their text.
    Returns a list of segments with speaker, start_time, end_time, and text.
    """
    segments = []
    lines = srt_text.strip().split('\n')
    
    for line in lines:
        # Match pattern with periods instead of commas, and optional arrow
        match = re.match(r'^\d+\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+SPEAKER_(\d+):\s*(.*)$', line)
        
        if match:
            # Extract start time
            start_h, start_m, start_s, start_ms = map(int, match.groups()[0:4])
            total_start_seconds = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000.0
            start_time = format_time_mm_ss(total_start_seconds)
            
            # Extract end time
            end_h, end_m, end_s, end_ms = map(int, match.groups()[4:8])
            total_end_seconds = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000.0
            end_time = format_time_mm_ss(total_end_seconds)
            
            # Extract speaker number and text
            speaker_num = int(match.group(9))
            text = match.group(10).strip()
            
            segments.append({
                'speaker': speaker_num,
                'start': start_time,
                'end': end_time,
                'text': text
            })
    
    return segments

def assign_speakers_to_tokens(tokens, speaker_segments):
    """
    Assign speaker numbers and timestamps to tokens based on time distribution.
    Distributes tokens evenly across the total duration, then assigns speakers based on time.
    """
    if not tokens or not speaker_segments:
        return tokens
    
    # Get start time from first segment
    first_segment = speaker_segments[0]
    first_start_minutes = int(first_segment['start'])
    first_start_seconds = (first_segment['start'] - first_start_minutes) * 100
    start_time_seconds = first_start_minutes * 60 + first_start_seconds
    
    # Get end time from last segment
    last_segment = speaker_segments[-1]
    last_end_minutes = int(last_segment['end'])
    last_end_seconds = (last_segment['end'] - last_end_minutes) * 100
    total_end_seconds = last_end_minutes * 60 + last_end_seconds
    
    # Calculate duration
    total_duration_seconds = total_end_seconds - start_time_seconds
    
    print(f"\nStart time: {start_time_seconds:.2f} seconds")
    print(f"End time: {total_end_seconds:.2f} seconds")
    print(f"Total duration: {total_duration_seconds:.2f} seconds ({total_duration_seconds/60:.2f} minutes)")
    print(f"Total tokens: {len(tokens)}")
    print(f"Time per token: {total_duration_seconds/len(tokens):.3f} seconds")
    
    # Distribute tokens evenly across time
    time_per_token = total_duration_seconds / len(tokens)
    
    for i, token in enumerate(tokens):
        # Calculate this token's time in seconds (starting from first segment)
        token_time_seconds = start_time_seconds + (i * time_per_token)
        
        # Convert to MM.SS format
        token_time = format_time_mm_ss(token_time_seconds)
        
        # Find which speaker segment this time falls into
        assigned_speaker = None
        for segment in speaker_segments:
            seg_start_min = int(segment['start'])
            seg_start_sec = (segment['start'] - seg_start_min) * 100
            seg_start_total = seg_start_min * 60 + seg_start_sec
            
            seg_end_min = int(segment['end'])
            seg_end_sec = (segment['end'] - seg_end_min) * 100
            seg_end_total = seg_end_min * 60 + seg_end_sec
            
            if seg_start_total <= token_time_seconds <= seg_end_total:
                assigned_speaker = segment['speaker']
                break
        
        # If no segment found (shouldn't happen), use last segment's speaker
        if assigned_speaker is None:
            assigned_speaker = speaker_segments[-1]['speaker']
        
        token['speaker'] = assigned_speaker
        token['start'] = round(token_time, 2)
        token['end'] = round(token_time, 2)
    
    print(f"\n✓ All {len(tokens)} tokens assigned")
    return tokens

def main(tokens_file_path, srt_file_path, output_file_path):
    """
    Assign speakers and timestamps to pre-generated tokens.
    
    Args:
        tokens_file_path: Path to JSON file with incomplete tokens
        srt_file_path: Path to SRT file with speaker data
        output_file_path: Path for output JSON
    """
    print("Loading tokens...")
    with open(tokens_file_path, 'r', encoding='utf-8') as f:
        tokens = json.load(f)
    print(f"✓ Loaded {len(tokens)} tokens")
    
    print("\nLoading SRT data...")
    with open(srt_file_path, 'r', encoding='utf-8') as f:
        srt_text = f.read()
    
    print("\nParsing speaker segments...")
    speaker_segments = parse_srt_speakers(srt_text)
    print(f"✓ Found {len(speaker_segments)} segments")
    
    print("\nAssigning speakers to tokens...")
    completed_tokens = assign_speakers_to_tokens(tokens, speaker_segments)
    
    print("\nSaving output...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(completed_tokens, f, indent=4)
    
    print(f"\n✓ Done! Saved to: {output_file_path}")
    print(f"  Total tokens: {len(completed_tokens)}")

if __name__ == "__main__":
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'i']
    
    for file_letter in files:
        tokens_file = rf"C:\Users\nathanjruhmann\Documents\stripped_SRT\rawtokens\incomplete_tokens_{file_letter}.json"
        srt_file = rf"C:\Users\nathanjruhmann\diarscription\docs\reference\audio\sample-{file_letter}\formatted_srt.md"
        output_file = rf"C:\Users\nathanjruhmann\Documents\stripped_SRT\tokens\completed_tokens_{file_letter}.json"
        
        main(tokens_file, srt_file, output_file)