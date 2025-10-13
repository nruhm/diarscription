import os
import tempfile
import subprocess
import sys
import stat
import urllib.request
import shutil

# Converts audio file to 16kHz mono WAV using FFmpeg. This is required for whisperx preprocessing. 

def setup_ffmpeg():
    temp_dir = tempfile.gettempdir()
    ffmpeg_path = os.path.join(temp_dir, "ffmpeg")

    if os.path.exists(ffmpeg_path):
        return ffmpeg_path

    if shutil.which("ffmpeg"):
        return "ffmpeg"

    try:
        if sys.platform.startswith('linux'):
            url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
            archive_path = os.path.join(temp_dir, "ffmpeg.tar.xz")
            urllib.request.urlretrieve(url, archive_path)
            subprocess.run(["tar", "-xf", archive_path, "-C", temp_dir], check=True)
            extracted_dir = next(d for d in os.listdir(temp_dir) if d.startswith("ffmpeg-master"))
            shutil.move(os.path.join(temp_dir, extracted_dir, "bin", "ffmpeg"), ffmpeg_path)
            os.chmod(ffmpeg_path, stat.S_IRWXU)
            os.remove(archive_path)
            shutil.rmtree(os.path.join(temp_dir, extracted_dir))
            return ffmpeg_path
        else:
            raise Exception("Unsupported platform")
    except Exception as e:
        raise Exception(f"Failed to setup FFmpeg: {e}")

def preprocess_audio(input_file, ffmpeg_path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tempfile.gettempdir()) as temp_file:
        temp_path = temp_file.name

    try:
        result = subprocess.run([
            ffmpeg_path, "-i", input_file, "-ar", "16000", "-ac", "1", "-y", temp_path
        ], capture_output=True, text=True, check=True)
        return temp_path
    except subprocess.CalledProcessError as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise Exception(f"FFmpeg failed: {e.stderr}")