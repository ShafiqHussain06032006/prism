"""
config_schema.py — Configuration parameters schema and validation.
"""
from typing import TypedDict, Optional

class PrismConfig(TypedDict, total=False):
    base_storage_path: str
    ali_dashscope_api_key: Optional[str]
    openai_api_key: Optional[str]
    gemini_api_key: Optional[str]
    image_bind_model_path: Optional[str]
