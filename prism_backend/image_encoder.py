"""
image_encoder.py — Convert PIL Images or files into base64 JPEG format.
"""
import base64
from io import BytesIO
from PIL import Image

def encode_pil_to_b64(image: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
