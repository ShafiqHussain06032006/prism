"""
prompt_templates.py — Multimodal Vision LLM prompt templates.
"""
DEFAULT_VISION_QA_PROMPT = """
Refer to the provided video frame image and associated transcript segment to answer the user question accurately.
Transcript segment: "{transcript}"
Question: {query}
"""
