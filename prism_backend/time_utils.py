"""
time_utils.py — Format timestamps into human readable strings.
"""
def format_seconds_to_timestamp(total_seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    mins, secs = divmod(int(total_seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"
