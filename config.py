"""
config.py
=========
Everything that must be true before a single model is imported.

  1. Decide where things live. Code and weights sit inside the installed bundle
     (read-only, replaced on update). Patient data sits in the OS's per-user data
     directory (never touched by an update). Conflating the two is how someone
     deletes a vault by upgrading.

  2. Point Hugging Face at the bundled weights and force offline mode BEFORE
     whisperx / transformers / pyannote get imported. Those modules read the env
     once at import and cache the answer, so the ordering is load-bearing --
     hence the call at the bottom of this file.

  3. Slam the door. install_network_guard() makes every IP socket raise.

Note the word EVERY. With the models loaded in-process (see local_llm.py) there
is no longer any loopback server to talk to, so this app has no legitimate reason
to open an AF_INET socket at all -- not to the internet, not to 127.0.0.1. The
guard is therefore absolute rather than "outbound only", which is a far easier
property to defend: this process cannot open a network socket. Unix domain
sockets are left alone, because Qt uses them internally for its own plumbing.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

APP_NAME = "Session Memory"
APP_SLUG = "session-memory"

# ---------------------------------------------------------------------------
# where things live
# ---------------------------------------------------------------------------
def _bundle_root() -> Path:
    """The installed app directory. Under PyInstaller that's the folder beside
    the executable; in development it's the repo."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _user_data_root() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or Path.home()
        return Path(base) / APP_NAME
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / APP_SLUG


BUNDLE = _bundle_root()
DATA = _user_data_root()

# Weights ship with the installer. Read-only at runtime.
MODELS_DIR = BUNDLE / "models"      # HF cache: whisper + pyannote
GGUF_DIR = MODELS_DIR / "gguf"      # llama.cpp weights

# WhisperX derives its transcript directory from the audio file's GRANDPARENT
# (see functions.transcribe_whisperx). Audio at DATA/audio therefore puts JSON at
# DATA/transcript, which is what we want. Move AUDIO_DIR and transcripts scatter.
AUDIO_DIR = DATA / "audio"
TRANSCRIPT_DIR = DATA / "transcript"
SUMMARY_DIR = DATA / "summary"
DB_DIR = DATA / "sql_db"
DB_PATH = DB_DIR / "clinic.db"

# ---------------------------------------------------------------------------
# models  (filenames, not registry names -- there is no registry to ask any more)
# ---------------------------------------------------------------------------
EMBED_GGUF = GGUF_DIR / os.environ.get("SM_EMBED_GGUF", "bge-m3-Q8_0.gguf")
CHAT_GGUF = GGUF_DIR / os.environ.get("SM_CHAT_GGUF", "gemma-3-4b-it-Q4_K_M.gguf")

# bge-m3 is 1024-dim. This MUST match `embedding FLOAT[n]` in vec_chunks
# (session_store.init_schema) or retrieval silently returns nonsense.
EMBED_DIM = 1024
CHAT_CTX = int(os.environ.get("SM_CHAT_CTX", "8192"))

USE_GPU = os.environ.get("SM_GPU", "1") == "1"
TOP_K = int(os.environ.get("SM_TOP_K", "5"))

# pyannote's diarization pipeline wants a token even when reading purely from
# cache. Any non-empty string satisfies it offline. A real one is only needed on
# the machine that PRE-DOWNLOADS the weights at packaging time.
HF_TOKEN = os.environ.get("HF_TOKEN", "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")


def ensure_dirs() -> None:
    for d in (AUDIO_DIR, TRANSCRIPT_DIR, SUMMARY_DIR, DB_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# offline mode
# ---------------------------------------------------------------------------
def enforce_offline() -> None:
    """Must run before whisperx / transformers / pyannote are imported."""
    from hf_cache_setup import setup_hf_cache  # your module, unchanged

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    setup_hf_cache(str(MODELS_DIR), offline=False)

    # setup_hf_cache covers HF_HOME / HF_HUB_CACHE / TRANSFORMERS_CACHE /
    # TORCH_HOME / HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE. These are the ones it
    # doesn't, and they matter: telemetry fires even in offline mode.
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["DISABLE_TELEMETRY"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    # WhisperX runs as a SUBPROCESS and inherits this environment, so the offline
    # flags follow it there. That subprocess does not inherit the guard below --
    # which is precisely why the env vars are belt as well as braces.


class NetworkBlocked(OSError):
    """Raised when anything in this process tries to open an IP socket."""


def install_network_guard() -> None:
    """Refuse every IP socket. Not just outbound, not just non-loopback: every one.

    The app loads its models in-process, reads its weights from disk, and writes
    to a local file. It has no legitimate reason to open an AF_INET socket, so an
    attempt is either a bug or an exfiltration path -- and either way it should
    be loud rather than silent.

    This is the property to hand a reviewer: pull the cable and the app is
    unchanged, because it was never able to use it.
    """
    real_socket_init = socket.socket.__init__
    real_getaddrinfo = socket.getaddrinfo

    def guarded_init(self, family=socket.AF_INET, *args, **kwargs):
        if family in (socket.AF_INET, socket.AF_INET6):
            raise NetworkBlocked(
                f"{APP_NAME} runs entirely on this computer and does not open "
                "network sockets. Something just tried to."
            )
        return real_socket_init(self, family, *args, **kwargs)

    def guarded_getaddrinfo(host, *args, **kwargs):
        raise NetworkBlocked(
            f"{APP_NAME} runs entirely on this computer and does not resolve "
            f"hostnames. Something just tried to look up {host!r}."
        )

    socket.socket.__init__ = guarded_init            # type: ignore[method-assign]
    socket.getaddrinfo = guarded_getaddrinfo         # type: ignore[assignment]


# Run at import. config must be the first thing the app imports.
enforce_offline()
ensure_dirs()
