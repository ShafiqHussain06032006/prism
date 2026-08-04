"""
device_detector.py — Auto-detect compute hardware (MPS Metal GPU vs CUDA vs CPU).
"""
import torch

def get_compute_device() -> str:
    """Return mps for Apple Silicon GPU, cuda for Nvidia GPU, else cpu."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
