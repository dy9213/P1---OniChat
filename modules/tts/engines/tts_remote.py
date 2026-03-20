"""VOICEVOX remote TTS engine via HTTP."""
import time
import httpx

AVAILABLE: bool = True  # always available; requires endpoint to be configured


async def voicevox_tts(client: httpx.AsyncClient, text: str, speaker: int, endpoint: str) -> bytes:
    """Synthesize speech via VOICEVOX HTTP API."""
    base = endpoint.rstrip("/")
    t0 = time.monotonic()
    r1 = await client.post(f"{base}/audio_query", params={"text": text, "speaker": speaker})
    r1.raise_for_status()
    t1 = time.monotonic()
    print(f"[voicevox] audio_query: {(t1-t0)*1000:.0f}ms", flush=True)

    r2 = await client.post(f"{base}/synthesis", params={"speaker": speaker}, json=r1.json())
    r2.raise_for_status()
    t2 = time.monotonic()
    print(f"[voicevox] synthesis: {(t2-t1)*1000:.0f}ms  total: {(t2-t0)*1000:.0f}ms", flush=True)
    return r2.content
