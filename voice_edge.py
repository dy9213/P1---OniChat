# scp /Users/davin/Projects/GitHub/ai-manager-1/services/voice_edge/voice_edge.py dy9213@SmartMirror:/home/dy9213/VoiceEdge/voice_edge.py
# start process: pulseaudio --load=module-native-protocol-tcp --exit-idle-time=-1 --daemon
#keep terminal open

import os, time, wave, threading, requests, pyaudio, onnxruntime, uvicorn, tempfile, yaml, queue, json, subprocess, re
import numpy as np
from ctypes import *
from fastapi import FastAPI
from openwakeword.model import Model
from openai import OpenAI
from contextlib import asynccontextmanager

with open("../server_config.yaml") as f:
    server_cfg = yaml.safe_load(f)

# MagicMirror Push
MM_PUSH = False
LED_THINKING = False

# FastAPI Config
VE_TRIGGER_IN = "/listen"

# Manager Server
STT_URL = f"http://{server_cfg['STT_URL']}:{server_cfg['STT_PORT']}/v1" #"http://davin-9950x:5006/v1"
STT_KEY = "local"

MANAGER_URL = f"http://{server_cfg['MANAGER_URL']}:{server_cfg['MANAGER_PORT']}"
MANAGER_INT = f"{MANAGER_URL}{server_cfg['MANAGER_INT']}"
MANAGER_STR = f"{MANAGER_URL}{server_cfg['MANAGER_STR']}"
MM_URL = f"http://{server_cfg['MM_URL']}:{server_cfg['MM_PORT']}"
MM_NOTIFY = f"{MM_URL}{server_cfg['MM_NOTIFY']}"

print(' -- SERVER CONFIG ----------------------- ',
        '\n> STT_URL:', STT_URL,
        '\n> MANAGER_URL:', MANAGER_URL,
        '\n> MANAGER_INT:', MANAGER_INT,
        '\n> MANAGER_STR:', MANAGER_STR,
        '\n> MM_URL:', MM_URL,
        '\n> MM_NOTIFY:', MM_NOTIFY,
        '\n-----------------------------------------'
    )


# Module Config
try:
    PLATFORM = os.getenv("PLATFORM", "mac") 
except:
    PLATFORM = "mac"

CHANNELS = 1
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "silero_vad.onnx")

if PLATFORM == "mac":
    IS_MAC = True
    RATE = 16000
    CHUNK_SIZE = 1280
    INPUT_DEVICE = None
else:
    # Raspberry Pi settings
    IS_MAC = False
    RATE = 48000
    CHUNK_SIZE = 3840
    INPUT_DEVICE = 2
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)


# Thresholds
NO_SPEECH_TIMEOUT = 5.0
SILENCE_THRESHOLD = 0.5 #Time: Second
RMS_THRESHOLD = 200
OWW_THRESHOLD = 0.6
MAX_DURATION = 30.0 #Time: Second

# --- SHARED STATE ---
class SystemState:
    def __init__(self):
        self.is_listening = False
        self.is_busy = False
        self.trigger_event = threading.Event()

state = SystemState()

# --- VOICE ENGINE ---
class VoiceEngine(threading.Thread):

    def __init__(self):
        super().__init__(daemon=True)
        self.audio = pyaudio.PyAudio()
        self.stt_client = OpenAI(base_url=STT_URL, api_key=STT_KEY)
        self.orig_volume = None
        self.oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
        self.vad_session = onnxruntime.InferenceSession(MODEL_PATH)
        self._reset_vad()
        self.state_queue = queue.Queue()
        threading.Thread(target=self.listening_worker, daemon=True).start()
        print(f"⏳ Loading Models ({'Mac' if IS_MAC else 'Linux'})...")

    def led_listen(self, listening):
        if listening:
            res = subprocess.check_output(["amixer", "-c", "3", "sget", "PCM"]).decode()
            self.orig_volume = re.search(r'\[(\d+)%\]', res).group(1)
            subprocess.run(["amixer", "-c", "3", "sset", "PCM", "50%"], capture_output=True)
        
        else:
            subprocess.run(["amixer", "-c", "3", "sset", "PCM", f"{self.orig_volume}%"], capture_output=True)

    def listening_worker(self):
        while True:
            current_status = self.state_queue.get()
            
            if current_status == True:
                p_json = json.dumps({"title": "JARVIS", "message": "LISTENING", "timer": 5000})
                full_url = f"{MM_NOTIFY}?action=NOTIFICATION&notification=SHOW_ALERT&payload={p_json}"
            
            if current_status == False:
                full_url = f"{MM_NOTIFY}?action=NOTIFICATION&notification=HIDE_ALERT"

            if current_status != None:
                try:
                    print(f"JARVIS LISTENING: {current_status}")
                    if MM_PUSH:
                        resp = requests.get(full_url, timeout=2)
    
                    if LED_THINKING:
                        self.led_listen(current_status)
                        time.sleep(1)
                    
                except Exception as e:
                    print(f"API Request failed: {e}")

            self.state_queue.task_done()

    def _reset_vad(self):
        self.vad_state = np.zeros((2, 1, 128), dtype=np.float32)

    def get_speech_prob(self, chunk_512):
        audio_f32 = chunk_512.astype(np.float32) / 32768.0
        out, self.vad_state = self.vad_session.run(None, {
            'input': audio_f32[np.newaxis, :], 
            'sr': np.array([16000], dtype=np.int64), # VAD always expects 16k
            'state': self.vad_state
        })
        return out[0][0]

    def check_vad(self, chunk_16k):
        """Processes a 16kHz resampled chunk for speech."""
        p1 = self.get_speech_prob(chunk_16k[0:512])
        p2 = self.get_speech_prob(chunk_16k[512:1024])
        return max(p1, p2)

    def trigger_interrupt(self):
        print(f"(SST -> MANAGER: INTERRUPT)")
        def send():
            try: requests.post(MANAGER_INT, timeout=1.0)
            except Exception as e: print(f"❌ SST -> MANAGER Error: {e}")

        threading.Thread(target=send, daemon=True).start()

    def send_string(self, string):
        print(f"(SST -> MANAGER: '{string}')")
        try: requests.post(MANAGER_STR, json={"sst": str(string)}, timeout=1.0)
        except Exception as e: print(f"❌ SST -> MANAGER Error: {e}")

    def run(self):
        stream = self.audio.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, 
                                 input=True, input_device_index=INPUT_DEVICE, 
                                 frames_per_buffer=CHUNK_SIZE)
        print(f"\n✅ Jarvis Online.")
        waiting_printed = False

        while True:
            if not state.is_busy and not state.is_listening:
                if not waiting_printed:
                    print("Waiting for Wake Word")
                    waiting_printed = True
            else:
                waiting_printed = False

            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # --- RESAMPLE FOR MODELS (48k -> 16k) ---
            audio_16k = audio_data[::3] if not IS_MAC else audio_data

            if self.oww_model.predict(audio_16k)["hey_jarvis"] > OWW_THRESHOLD or state.trigger_event.is_set():
                state.is_busy = True
                state.trigger_event.clear()
                waiting_printed = False
                
                self.trigger_interrupt() #always send interrupt
                time.sleep(0.5)
                self.record_and_process(stream)
                self.oww_model.reset()
                self._reset_vad()

    def record_and_process(self, stream):
        state.is_listening, frames, silence_start = True, [], None
        max_rms = 0
        self.state_queue.put(True)
        self.is_listening = True

        record_start = time.time()
        speech_detected = False  # Track if the user has actually started talking
        
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            frames.append(data)

            # Record Max RMS
            chunk_rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
            if chunk_rms > max_rms:
                max_rms = chunk_rms

            # 1. Global Safety Timeout
            if (time.time() - record_start) > MAX_DURATION:
                print(f"⏱️ Timeout: Reached {MAX_DURATION}s Limit.")
                stop_time = time.perf_counter()
                break

            # Resample slice for VAD check
            check_slice = audio_data[::3] if not IS_MAC else audio_data
            
            # 2. VAD Logic
            if self.check_vad(check_slice) > 0.6:
                speech_detected = True # User started talking!
                silence_start = None
            else:
                if silence_start is None: 
                    silence_start = time.time()
                
                # 3. Check for specific silence conditions
                elapsed_silence = time.time() - silence_start
                
                if not speech_detected and elapsed_silence > NO_SPEECH_TIMEOUT:
                    print("🔇 No speech detected within 5 seconds. Terminating.")
                    stop_time = time.perf_counter()
                    break
                elif speech_detected and elapsed_silence > SILENCE_THRESHOLD:
                    # This handles the normal "done talking" pause
                    stop_time = time.perf_counter()
                    break
            
            if len(frames) * CHUNK_SIZE / RATE > 15: break
        #print("Listening: Stopped")
        self.state_queue.put(False)
        self.is_listening = False

        if max_rms < RMS_THRESHOLD:
            print(f"VE Error: Too Quiet ({max_rms:.1f})")
            state.is_busy = False
            return
        
        threading.Thread(target=self.process_pipeline, args=(b''.join(frames),stop_time)).start()

    def process_pipeline(self, audio_data, stop_time):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                with wave.open(tf.name, 'wb') as f:
                    f.setnchannels(CHANNELS); f.setsampwidth(2); f.setframerate(RATE); f.writeframes(audio_data)
                audio_path = tf.name

            send_start_time = time.perf_counter()
            vad_to_send_latency = send_start_time - stop_time
            print(f"Latency (VAD -> Network Send): {vad_to_send_latency:.3f}s")


            with open(audio_path, "rb") as f:
                    try: txt = self.stt_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        language=server_cfg['STT_LANG'],
                        ).text

                    except Exception as e: 
                        print(f"❌ STT Error: {e}")
                        txt = 'Sorry, Connection Error'

            if not txt.strip(): return
            self.send_string(txt)

        except Exception as e: 
            print(f"❌ STT Error: {e}")

        finally:
            state.is_busy = False
            if os.path.exists(audio_path): os.remove(audio_path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    VoiceEngine().start()
    yield

app = FastAPI(lifespan=lifespan)

@app.post(VE_TRIGGER_IN)
async def manual_listen():

    if state.is_busy:
        current_status = {
            "state": "busy"
        } 

    else:
        current_status = {
            "state": "listening"
        }

        state.trigger_event.set()

    return current_status

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=server_cfg['VE_PORT'])