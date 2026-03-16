"""
Local MLX-based Qwen3-TTS module.
Import this in main.py; it loads the model once at import time.

Usage:
    from tts_mlx import mlx_tts_stream
    async for pcm_chunk in mlx_tts_stream(text, voice, language):
        await ws.send_bytes(pcm_chunk)

Each yielded chunk is raw float32 PCM bytes at SAMPLE_RATE (24000 Hz).
"""

import asyncio
import logging
import os

import numpy as np

log = logging.getLogger("tts_mlx")

MODEL_ID    = os.getenv("MLX_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit")
SAMPLE_RATE = 24000   # Qwen3-TTS native output sample rate

_model = None

def _load():
    global _model
    if _model is not None:
        return
    try:
        from mlx_audio.tts import load
        _model = load(MODEL_ID)
        log.info(f"[tts_mlx] loaded {MODEL_ID}")
    except Exception as e:
        log.error(f"[tts_mlx] failed to load model: {e}")
        raise


def _is_available() -> bool:
    try:
        import mlx_audio  # noqa
        return True
    except ImportError:
        return False


AVAILABLE = _is_available()

if AVAILABLE:
    try:
        _load()
    except Exception:
        AVAILABLE = False


async def mlx_tts_stream(text: str, voice: str, language: str):
    """
    Async generator — yields raw float32 PCM chunks.
    Runs the blocking inference in a thread pool so it doesn't block the event loop.
    `language` is accepted for API compatibility but Qwen3-TTS infers it from the text.
    """
    if not AVAILABLE or _model is None:
        raise RuntimeError("MLX TTS not available")

    loop = asyncio.get_event_loop()

    def _generate():
        # generate() is a Python generator; collect results in the thread
        return list(_model.generate(
            text=text,
            voice=voice,
            stream=True,
            streaming_interval=1.0,
        ))

    results = await loop.run_in_executor(None, _generate)

    for result in results:
        arr = np.asarray(result.audio, dtype=np.float32)
        yield arr.tobytes()
        # yield control between chunks so WebSocket send can interleave
        await asyncio.sleep(0)
