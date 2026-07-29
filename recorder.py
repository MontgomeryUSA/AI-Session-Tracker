"""
recorder.py
===========
Microphone -> 16 kHz mono PCM WAV, written incrementally to disk.

Why not just grab audio and convert later: WhisperX wants 16 kHz mono, and the
old browser path recorded webm/opus and then shelled out to ffmpeg to fix it.
Recording in the target format in the first place deletes that whole step, the
ffmpeg dependency, and an entire class of "the codec was wrong" bugs.

Frames stream to disk as they arrive rather than piling up in RAM, so a
90-minute session costs ~170 MB of disk and almost no memory, and a crash leaves
a valid, playable partial WAV instead of nothing.

Most interfaces will happily open at 16 kHz. The ones that won't (some USB
interfaces are fixed at 44.1/48 kHz) raise, and we reopen at the device's own
rate and resample on the way out -- see pipeline.prepare_audio.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

TARGET_SR = 16000
TARGET_CH = 1


@dataclass
class Device:
    index: int
    name: str
    default: bool


def input_devices() -> list[Device]:
    default_idx = sd.default.device[0]
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append(Device(index=i, name=d["name"], default=(i == default_idx)))
    return out


class Recorder:
    """Start it, poll .level for the meter, stop it, get a path back.

    on_level: called from the audio thread with 0.0-1.0 RMS. Keep it cheap, and
    do not touch Qt widgets from inside it -- the UI marshals it across.
    """

    def __init__(self, on_level=None):
        self._on_level = on_level
        self._q: queue.Queue = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._writer: threading.Thread | None = None
        self._file: sf.SoundFile | None = None
        self._stop = threading.Event()
        self.path: Path | None = None
        self.samplerate: int = TARGET_SR
        self.frames: int = 0

    # -- audio thread -------------------------------------------------------
    def _callback(self, indata, frames, time_info, status):
        self._q.put(indata.copy())
        if self._on_level is not None:
            rms = float(np.sqrt(np.mean(np.square(indata, dtype=np.float64))))
            # Speech sits low in a linear RMS scale; a cube root opens it out so
            # the meter actually moves at conversational volume.
            self._on_level(min(1.0, (rms ** (1 / 3)) * 1.6))

    # -- writer thread ------------------------------------------------------
    def _drain(self):
        while not (self._stop.is_set() and self._q.empty()):
            try:
                block = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._file is not None:
                self._file.write(block)
                self.frames += len(block)

    # -- api ----------------------------------------------------------------
    def start(self, path: Path, device: int | None = None) -> None:
        if self._stream is not None:
            raise RuntimeError("Already recording.")
        self.path = Path(path)
        self.frames = 0
        self._stop.clear()

        try:
            self.samplerate = TARGET_SR
            stream = sd.InputStream(
                samplerate=TARGET_SR, channels=TARGET_CH, dtype="float32",
                device=device, callback=self._callback, blocksize=1024,
            )
        except Exception:
            # Device refused 16 kHz. Take whatever it offers; pipeline resamples.
            info = sd.query_devices(device if device is not None else sd.default.device[0])
            self.samplerate = int(info["default_samplerate"])
            stream = sd.InputStream(
                samplerate=self.samplerate, channels=TARGET_CH, dtype="float32",
                device=device, callback=self._callback, blocksize=1024,
            )

        self._file = sf.SoundFile(
            str(self.path), mode="w", samplerate=self.samplerate,
            channels=TARGET_CH, subtype="PCM_16",
        )
        self._writer = threading.Thread(target=self._drain, daemon=True)
        self._writer.start()

        self._stream = stream
        self._stream.start()

    def stop(self) -> Path:
        if self._stream is None:
            raise RuntimeError("Not recording.")
        self._stream.stop()
        self._stream.close()
        self._stream = None

        self._stop.set()
        if self._writer:
            self._writer.join(timeout=5)
        if self._file:
            self._file.close()
            self._file = None

        assert self.path is not None
        return self.path

    @property
    def seconds(self) -> float:
        return self.frames / self.samplerate if self.samplerate else 0.0

    @property
    def recording(self) -> bool:
        return self._stream is not None
