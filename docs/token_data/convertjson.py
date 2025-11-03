import json

with open(r'C:\Users\nathanjruhmann\diarscription\docs\reference\audio\sample-a\rawtoken.json', 'r') as f:
    data = json.load(f)

with open(r'C:\Users\nathanjruhmann\diarscription\docs\reference\audio\sample-a\tokendata.json', 'w') as f:
    f.write('[\n')
    for i, item in enumerate(data):
        f.write('    {\n')
        f.write(f'        "token": "{item["token"]}",\n')
        f.write(f'        "id": {item["id"]},\n')
        f.write(f'        "speaker": {item["speaker"]},\n')
        f.write(f'        "start": {item["start"]:.2f},\n')
        f.write(f'        "end": {item["end"]:.2f}\n')
        f.write('    }')
        if i < len(data) - 1:
            f.write(',\n')
        else:
            f.write('\n]')