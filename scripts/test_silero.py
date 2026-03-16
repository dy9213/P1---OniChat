"""
Standalone Silero VAD test — no Electron, no WebSocket, no AudioWorklet.
Matches voice_edge.py Mac path: request 16000 Hz directly from PyAudio (macOS Core Audio
handles SRC), no decimation, CHUNK_SIZE=1280 (two 512-sample Silero windows per read).

Usage:
    cd <project_root>
    pip install pyaudio onnxruntime numpy
    python scripts/test_silero.py

Speak after "Recording…" and watch prob values. Should exceed 0.6 during speech.
"""

import time
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyaudio

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = Path(__file__).parent.parent / "data" / "silero_vad.onnx"
MODEL_URL    = "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx"
SAMPLE_RATE  = 16000        # request 16kHz; macOS Core Audio resamples transparently
CHUNK        = 512          # samples per Silero window (32ms at 16kHz)
PYAUDIO_BUF  = 1280         # same as voice_edge.py CHUNK_SIZE for Mac — two Silero windows
THRESHOLD    = 0.6
DURATION_S   = 10           # record for this many seconds then stop

# ── model ─────────────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    print("Downloading Silero VAD model…")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
print("Model inputs:")
for i in session.get_inputs():
    print(f"  {i.name}  shape={i.shape}  dtype={i.type}")
print("Model outputs:")
for o in session.get_outputs():
    print(f"  {o.name}  shape={o.shape}  dtype={o.type}")

state = np.zeros((2, 1, 128), dtype=np.float32)

def infer(samples_int16: np.ndarray):
    """Run one 512-sample window. samples_int16: int16 array of length 512."""
    global state
    audio_f32 = samples_int16.astype(np.float32) / 32768.0
    out = session.run(None, {
        "input": audio_f32[np.newaxis, :],
        "sr":    np.array([SAMPLE_RATE], dtype=np.int64),
        "state": state,
    })
    state = out[1]
    return float(out[0][0][0])

# ── record ────────────────────────────────────────────────────────────────────
pa = pyaudio.PyAudio()
dev_info = pa.get_default_input_device_info()
print(f"Default input device: {dev_info['name']}  native_rate={dev_info['defaultSampleRate']:.0f}")
print(f"Requesting {SAMPLE_RATE} Hz from PyAudio (Core Audio handles SRC)\n")

stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                 input=True, frames_per_buffer=PYAUDIO_BUF)

print(f"Recording for {DURATION_S}s — speak now…\n")
t_end   = time.time() + DURATION_S
win_num = 0
buf     = np.empty(0, dtype=np.int16)

try:
    while time.time() < t_end:
        raw   = stream.read(PYAUDIO_BUF, exception_on_overflow=False)
        chunk = np.frombuffer(raw, dtype=np.int16)
        buf   = np.concatenate([buf, chunk])

        while len(buf) >= CHUNK:
            window = buf[:CHUNK]
            buf    = buf[CHUNK:]
            win_num += 1
            rms  = float(np.sqrt(np.mean(window.astype(np.float32) ** 2)))
            prob = infer(window)
            tag  = "  ← SPEECH" if prob >= THRESHOLD else ""
            print(f"win#{win_num:03d}  rms={rms:.1f} (f32≈{rms/32768:.4f})  prob={prob:.4f}{tag}")

finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("\nDone.")
