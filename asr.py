import os
import asyncio
from .._utils import logger


# ---------------------------------------------------------------------------
# Gemini ASR (primary path when gemini_api_key is present)
# ---------------------------------------------------------------------------

async def _gemini_transcribe_segment(semaphore, index, segment_name, audio_file, api_key):
    """
    Transcribe a single audio segment using the Google Gemini Files API.
    The audio is uploaded to Gemini's file store and then the model is asked
    to return only the spoken transcript.
    """
    async with semaphore:
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=api_key)
            logger.info(f"🔊 Gemini ASR: uploading segment {segment_name}")

            loop = asyncio.get_event_loop()

            # Upload is synchronous; run in thread pool so we don't block the loop
            uploaded_file = await loop.run_in_executor(
                None,
                lambda: genai.upload_file(path=audio_file, mime_type="audio/mpeg"),
            )

            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "Transcribe the following audio clip verbatim. "
                "Return ONLY the spoken words with no additional commentary."
            )

            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content([prompt, uploaded_file]),
            )

            text = response.text.strip() if response.text else ""
            logger.info(f"✅ Gemini ASR done for segment {segment_name}")
            return index, text

        except Exception as e:
            logger.error(f"❌ Gemini ASR failed for segment {segment_name}: {str(e)}")
            return index, ""


async def speech_to_text_gemini(
    video_name, working_dir, segment_index2name, audio_output_format, global_config, max_concurrent=3
):
    """
    ASR using Google Gemini Files API. Activated when gemini_api_key is present
    in global_config.
    """
    api_key = (
        global_config.get("gemini_api_key")
        or os.environ.get("GEMINI_API_KEY", "")
    ).strip()

    if not api_key:
        raise ValueError("gemini_api_key is required for Gemini ASR")

    cache_path = os.path.join(working_dir, "_cache", video_name)
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = []
    for index, segment_name in segment_index2name.items():
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")
        tasks.append(
            _gemini_transcribe_segment(semaphore, index, segment_name, audio_file, api_key)
        )

    total = len(tasks)
    logger.info(f"🎤 Starting Gemini ASR for {total} audio segments (max {max_concurrent} concurrent)...")

    transcripts = {}
    completed = 0
    for coro in asyncio.as_completed(tasks):
        try:
            idx, text = await coro
            transcripts[idx] = text
        except Exception as e:
            logger.error(f"❌ Gemini ASR task failed: {e}")
        completed += 1
        logger.info(f"   Progress: {completed}/{total} ({completed/total*100:.1f}%)")

    logger.info(f"🎉 Gemini ASR completed! Processed {len(transcripts)} segments.")
    return transcripts


# ---------------------------------------------------------------------------
# DashScope ASR (legacy fallback)
# ---------------------------------------------------------------------------

async def _dashscope_transcribe_segment(semaphore, index, segment_name, audio_file, model, audio_output_format, sample_rate):
    """Process a single audio segment with DashScope ASR."""
    async with semaphore:
        try:
            import dashscope  # type: ignore
            from dashscope.audio.asr import Recognition  # type: ignore

            logger.info(f"Processing segment {segment_name} with model {model}")
            recognition = Recognition(
                model=model,
                format=audio_output_format,
                sample_rate=sample_rate,
                language_hints=["zh", "en", "ja"],
                callback=None,  # type: ignore
            )
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, recognition.call, audio_file)

            if result and "output" in result and "sentence" in result["output"]:
                sentences = result["output"]["sentence"]
                asr_result = "".join(s.get("text", "") + "\n" for s in sentences)
                return index, asr_result.strip()
            else:
                logger.warning(f"No transcription result for segment {segment_name}")
                return index, ""
        except Exception as e:
            logger.error(f"ASR failed for segment {segment_name}: {str(e)}")
            raise e


async def speech_to_text_dashscope(
    video_name, working_dir, segment_index2name, audio_output_format, global_config, max_concurrent=5
):
    """Online ASR using Alibaba Cloud DashScope API with async concurrent processing."""
    import dashscope  # type: ignore

    api_key = global_config.get("ali_dashscope_api_key")
    sample_rate = global_config.get("audio_sample_rate", 16000)
    dashscope.api_key = api_key

    cache_path = os.path.join(working_dir, "_cache", video_name)
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        _dashscope_transcribe_segment(
            semaphore, index, segment_index2name[index],
            os.path.join(cache_path, f"{segment_index2name[index]}.{audio_output_format}"),
            global_config.get("asr_model"), audio_output_format, sample_rate,
        )
        for index in segment_index2name
    ]

    total = len(tasks)
    logger.info(f"🎤 Starting DashScope ASR for {total} audio segments (max {max_concurrent} concurrent)...")

    transcripts = {}
    completed = 0
    for coro in asyncio.as_completed(tasks):
        try:
            idx, text = await coro
            transcripts[idx] = text
            completed += 1
            logger.info(f"✅ Completed {completed}/{total} segments ({completed/total*100:.1f}%)")
        except Exception as e:
            completed += 1
            logger.error(f"❌ Task failed: {e}")

    logger.info(f"🎉 DashScope ASR completed! Processed {len(transcripts)} segments.")
    return transcripts


# ---------------------------------------------------------------------------
# Public API — unchanged signatures, auto-routes to Gemini or DashScope
# ---------------------------------------------------------------------------

async def speech_to_text_async(
    video_name, working_dir, segment_index2name, audio_output_format, global_config
):
    """
    Async speech-to-text: routes to Gemini when gemini_api_key is present,
    otherwise falls back to DashScope.

    Args:
        video_name: Name of the video
        working_dir: Working directory
        segment_index2name: Mapping of segment indices to names
        audio_output_format: Audio file format
        global_config: Global configuration dictionary
    """
    gemini_key = (
        global_config.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY", "")
    ).strip()

    if gemini_key:
        logger.info("🌟 ASR: Using Google Gemini transcription")
        return await speech_to_text_gemini(
            video_name, working_dir, segment_index2name, audio_output_format, global_config
        )
    else:
        logger.info("🔵 ASR: Using DashScope (paraformer) transcription")
        dashscope_key = global_config.get("ali_dashscope_api_key")
        if not dashscope_key:
            raise ValueError(
                "ali_dashscope_api_key must be provided in global_config when gemini_api_key is not set"
            )
        return await speech_to_text_dashscope(
            video_name, working_dir, segment_index2name, audio_output_format, global_config
        )


def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format, global_config):
    """
    Synchronous wrapper for async speech-to-text function.

    Args:
        video_name: Name of the video
        working_dir: Working directory
        segment_index2name: Mapping of segment indices to names
        audio_output_format: Audio file format
        global_config: Global configuration dictionary
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        speech_to_text_async(video_name, working_dir, segment_index2name, audio_output_format, global_config)
    )