"""
Manage a voicevox_engine subprocess for local Japanese TTS.
Binary:  modules/tts/bin/run  (VOICEVOX Engine release binary)
Port:    50021 (VOICEVOX default, loopback only)
"""
import os, subprocess, threading, time, urllib.request
from pathlib import Path
from typing import Optional

_USER_DATA    = Path(os.environ.get("ONICHAT_USER_DATA", Path(__file__).parent.parent.parent))
BIN_DIR       = _USER_DATA / "modules" / "tts" / "bin"
VOICEVOX_PORT = 50021
VOICEVOX_URL  = f"http://127.0.0.1:{VOICEVOX_PORT}"

# VOICEVOX Engine binary candidates (checked in order)
_BIN_CANDIDATES = ["run", "voicevox_engine", "main"]


def _find_binary() -> Optional[Path]:
    for name in _BIN_CANDIDATES:
        p = BIN_DIR / name
        if p.exists() and os.access(p, os.X_OK):
            return p
    return None


class VoicevoxManager:
    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None

    def is_installed(self) -> bool:
        return _find_binary() is not None

    def is_running(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            return True
        # Also treat the engine as running if it's already healthy on the port
        # (e.g. a previous backend instance started it and we restarted).
        try:
            with urllib.request.urlopen(f"{VOICEVOX_URL}/speakers", timeout=1) as r:
                return r.status == 200
        except Exception:
            return False

    @property
    def endpoint(self) -> str:
        return VOICEVOX_URL

    def start(self) -> None:
        """Start voicevox_engine. Blocks until /speakers returns 200 (≤60 s).
        If already running but not owned by this manager, restart so the
        --enable_cancellable_synthesis flag is guaranteed to be active."""
        if self.is_running():
            if self._proc is not None:
                return  # we own it and it's running — already started with correct flags
            self.stop()  # orphan from a previous session — kill and restart

        binary = _find_binary()
        if binary is None:
            raise RuntimeError(
                "voicevox_engine binary not found — run the installer first"
            )

        def _boost_priority():
            """Raise VOICEVOX process nice value so inference isn't starved by
            lower-QoS process tree inherited from launchd / Finder launch."""
            try:
                os.setpriority(os.PRIO_PROCESS, 0, -5)
            except Exception:
                pass

        self._proc = subprocess.Popen(
            [str(binary), "--host", "127.0.0.1", "--port", str(VOICEVOX_PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd=BIN_DIR,   # engine expects to run from its own directory
            preexec_fn=_boost_priority,
        )

        # Forward VOICEVOX stderr to backend stdout for console diagnostics
        def _pipe_stderr(proc):
            for raw in proc.stderr:
                line = raw.decode(errors="replace").rstrip()
                if line:
                    print(f"[voicevox_engine] {line}", flush=True)

        threading.Thread(target=_pipe_stderr, args=(self._proc,), daemon=True).start()

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("voicevox_engine exited unexpectedly during startup")
            try:
                with urllib.request.urlopen(f"{VOICEVOX_URL}/speakers", timeout=3) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(1)

        self.stop()
        raise RuntimeError("voicevox_engine did not become healthy within 60 s")

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        else:
            # No owned process — kill any orphan holding the port
            try:
                import signal
                result = subprocess.run(
                    ["lsof", "-ti", f":{VOICEVOX_PORT}"],
                    capture_output=True, text=True,
                )
                for pid in result.stdout.strip().split():
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except Exception:
                        pass
            except Exception:
                pass
        self._proc = None

    def restart(self) -> None:
        self.stop()
        self.start()


voicevox_manager = VoicevoxManager()
