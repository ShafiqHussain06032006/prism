"""
logger.py — Timestamped thread-safe logger utility.
"""
import datetime

def log_event(message: str, log_file_path: str = None):
    """Write timestamped message to stdout and optional log file."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    print(line)
    if log_file_path:
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(line + "
")
