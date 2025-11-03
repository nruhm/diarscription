import tiktoken
import json

encoding = tiktoken.get_encoding("p50k_base")
files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'i']

for file_letter in files:
    # Read the stripped SRT file
    input_path = rf"C:\Users\nathanjruhmann\Documents\stripped_SRT\audio-{file_letter}.txt"
    
    with open(input_path, "r", encoding='utf-8') as f:
            srt_content = f.read()
    
    # Encode and decode with tiktoken
    encode = encoding.encode(srt_content)
    decode = encoding.decode(encode)
    
    individual_tokens = [encoding.decode_single_token_bytes(token) for token in encode]
    individual_tokens_str = [token.decode('utf-8') for token in individual_tokens]
    filtered_tokens = [token for token in individual_tokens_str if token.strip()]
    
    # Build array of token dictionaries
    tokens = []
    for i, token in enumerate(filtered_tokens):
        tokens.append({
            "token": token,
            "id": i,
            "speaker": None,
            "start": None,
            "end": None
        })
    
    # Write to JSON file
    output_path = rf"C:\Users\nathanjruhmann\Documents\stripped_SRT\rawtokens\incomplete_tokens_{file_letter}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tokens, f, indent=4)