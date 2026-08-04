"""
video_decoder.py — Robust video frame decoder with decord and pyav fallbacks.
"""
from typing import Optional

def load_video_decoder(video_path: str):
    """Try loading video using decord first, fallback to pyav on failure."""
    try:
        from pytorchvideo.data.encoded_video import EncodedVideo
        try:
            return EncodedVideo.from_path(video_path, decoder="decord")
        except Exception:
            return EncodedVideo.from_path(video_path, decoder="pyav")
    except ImportError:
        return None
