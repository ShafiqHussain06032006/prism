"""
audio_extractor.py — Extract audio track from video files using ffmpeg.
"""
import subprocess

def extract_audio(video_path: str, output_wav_path: str) -> bool:
    """Extract 16kHz mono WAV audio from input video file."""
    cmd = f"ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{output_wav_path}""
    res = subprocess.run(cmd, shell=True, capture_output=True)
    return res.returncode == 0
