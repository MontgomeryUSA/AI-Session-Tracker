"""
pipeline.py
===========
The seam between your two systems, unchanged in intent from the terminal
version: one call takes audio and leaves you with a session you can ask
questions about.

    audio
      -> 16 kHz mono                      (usually a no-op now: see below)
      -> WhisperX transcribe + diarize    functions.transcribe_whisperx
      -> merged, role-tagged turns        functions.merge_consecutive_speakers
      -> SOAP note                        local model
      -> chunk + embed + encrypted store  session_store.ingest_session

What changed for the desktop build:

  * recorder.py already writes 16 kHz mono, so prepare_audio() is a no-op for
    anything recorded in-app. It only does work for files the clinician drags in
    from elsewhere.
  * Resampling is done with soundfile + scipy, which are already in your
    dependency tree, instead of shelling out to ffmpeg. A packaged desktop app
    shouldn't require the user to have ffmpeg on their PATH. ffmpeg is still
    used as a last resort for containers libsndfile can't open (m4a, mp4), and
    if it isn't there we say so plainly instead of throwing FileNotFoundError.
  * Nothing prints to a terminal any more. Progress goes through the `log` and
    `stage` callbacks, which the UI wires to the progress panel.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import config  # first, always: sets offline env before whisperx is imported

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import local_llm
import session_store as store
from functions import (
    merge_consecutive_speakers,
    merged_to_prompt_text,
    transcribe_whisperx,
)

TARGET_SR = 16000

SOAP_SYS = (
    "Draft a SOAP note ONLY from the transcript. Do not invent symptoms, "
    "diagnoses, or quotes. Mark anything inferred as [inferred]. Leave a "
    "field blank rather than guess."
)

DEFAULT_ROLE_MAP = {"SPEAKER_00": "Patient", "SPEAKER_01": "Therapist"}

Log = Callable[[str], None]
Stage = Callable[[str], None]

STAGES = ["preparing", "transcribing", "summarizing", "indexing", "done"]


def _noop(*_a) -> None:
    pass


@dataclass
class SessionResult:
    session_id: int
    patient_id: int
    audio_path: str
    transcript_path: str
    summary_path: str
    soap_note: str
    transcript_text: str
    chunk_count: int
    duration: float


# ---------------------------------------------------------------------------
# 1. get to 16 kHz mono without needing ffmpeg
# ---------------------------------------------------------------------------
def prepare_audio(audio_path: Path, log: Log = _noop) -> Path:
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"There's no audio file at {audio_path}")

    try:
        info = sf.info(str(audio_path))
    except Exception:
        return _transcode(audio_path, log)

    if info.samplerate == TARGET_SR and info.channels == 1:
        log("Audio is already 16 kHz mono. Nothing to convert.")
        return audio_path

    out = audio_path.with_name(f"{audio_path.stem}_16k.wav")
    if out.exists():
        log(f"Reusing {out.name}")
        return out

    log(f"Resampling {info.samplerate} Hz / {info.channels} ch to 16 kHz mono…")
    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)  # downmix before resampling: cheaper, same result

    if sr != TARGET_SR:
        g = np.gcd(int(sr), TARGET_SR)
        mono = resample_poly(mono, TARGET_SR // g, sr // g)

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1.0:
        mono = mono / peak  # resample_poly can overshoot; clipping would be audible

    sf.write(str(out), mono, TARGET_SR, subtype="PCM_16")
    log(f"Wrote {out.name}")
    return out


def _transcode(audio_path: Path, log: Log) -> Path:
    """libsndfile couldn't open it -- m4a, mp4, some webm. Only path that needs ffmpeg."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            f"{audio_path.suffix or 'That file type'} needs ffmpeg to open, and ffmpeg "
            "isn't installed.\n\nRecord in the app, or convert the file to WAV or FLAC first."
        )
    out = audio_path.with_name(f"{audio_path.stem}_16k.wav")
    if out.exists():
        log(f"Reusing {out.name}")
        return out
    log(f"Converting {audio_path.name} with ffmpeg…")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", str(TARGET_SR),
         "-sample_fmt", "s16", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out


# ---------------------------------------------------------------------------
# 2. SOAP note
# ---------------------------------------------------------------------------
def generate_soap(transcript_text: str, log: Log = _noop) -> str:
    """Streams so the progress panel keeps moving on a long transcript."""
    messages = [
        {"role": "system", "content": SOAP_SYS},
        {"role": "user", "content": transcript_text},
    ]
    parts: list[str] = []
    line: list[str] = []
    for piece in local_llm.chat_stream(messages, temperature=0.2, max_tokens=1536):
        parts.append(piece)
        line.append(piece)
        if "\n" in piece:                      # log a line at a time, not a token
            log("".join(line).strip())
            line = []
    if line:
        log("".join(line).strip())
    return "".join(parts)


def save_soap(soap: str, stem: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = config.SUMMARY_DIR / f"{stem.removesuffix('_proc')}_summary_{stamp}.txt"
    path.write_text(soap, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 3. the whole run
# ---------------------------------------------------------------------------
def process_audio(
    audio_path: str | Path,
    passphrase: str,
    *,
    patient_id: int = 1,
    patient_alias: str = "PT-0001",
    role_map: Optional[dict] = None,
    log: Log = _noop,
    stage: Stage = _noop,
    cancelled: Callable[[], bool] = lambda: False,
) -> SessionResult:
    role_map = role_map or DEFAULT_ROLE_MAP
    original = Path(audio_path).resolve()

    stage("preparing")
    wav = prepare_audio(original, log)
    if cancelled():
        raise InterruptedError

    stage("transcribing")
    log(f"Transcribing and separating speakers ({'GPU' if config.USE_GPU else 'CPU'}).")
    log("This is the slow part — roughly real-time on CPU, much faster on GPU.")
    raw_path, proc_path = transcribe_whisperx(
        audio_file=str(wav),
        hf_token=config.HF_TOKEN,
        use_gpu=config.USE_GPU,
        min_speakers=2,
        max_speakers=len(role_map),
    )
    if cancelled():
        raise InterruptedError

    stage("summarizing")
    merged = merge_consecutive_speakers(Path(proc_path), role_map)
    transcript_text = merged_to_prompt_text(merged)
    log("Drafting the SOAP note.")
    soap = generate_soap(transcript_text, log)
    summary_path = save_soap(soap, Path(proc_path).stem)
    if cancelled():
        raise InterruptedError

    stage("indexing")
    log("Embedding and encrypting into the vault.")
    con = store.connect(str(config.DB_PATH), passphrase)
    try:
        store.init_schema(con)
        session_id = store.ingest_session(
            con,
            patient_id=patient_id,
            patient_alias=patient_alias,
            json_path=str(proc_path),
            soap_note_path=str(summary_path),
            audio_path=str(original),
            role_map=role_map,
        )
        chunk_count, duration = con.execute(
            'SELECT COUNT(*), COALESCE(MAX("end"), 0) FROM chunk WHERE session_id = ?',
            (session_id,),
        ).fetchone()
    finally:
        con.close()

    log(f"Session {session_id} is in the vault ({chunk_count} chunks).")
    stage("done")

    return SessionResult(
        session_id=session_id,
        patient_id=patient_id,
        audio_path=str(original),
        transcript_path=str(proc_path),
        summary_path=str(summary_path),
        soap_note=soap,
        transcript_text=transcript_text,
        chunk_count=chunk_count,
        duration=float(duration),
    )
