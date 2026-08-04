"""
cache_initializer.py — Initialize model caching and storage directories.
"""
import os

def init_storage_directories(base_path: str = "~/Documents/prism"):
    """Ensure storage directory structure exists."""
    expanded = os.path.expanduser(base_path)
    os.makedirs(expanded, exist_ok=True)
    return expanded
