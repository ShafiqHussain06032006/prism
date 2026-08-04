"""
session_store.py — Thread-safe atomic JSON read/write helpers.
"""
import os, json

def atomic_write_json(path: str, data: dict):
    """Write JSON file atomically using temporary file move."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.rename(tmp, path)
