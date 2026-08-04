"""
error_handler.py — Exception formatting and boundary handlers.
"""
import traceback

def format_error_stack(e: Exception) -> str:
    """Format complete exception traceback string."""
    return traceback.format_exc()
