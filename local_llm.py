"""
local_llm.py
============
Both models, loaded into this process. No server, no daemon, no socket.

This replaces Ollama. Ollama was never a *network* dependency -- it listened on
127.0.0.1, so nothing left the machine -- but it was a second process, a TCP
port, and something the clinician had to have installed. For an app whose whole
premise is "it runs on your laptop and nothing else", that's the wrong shape.

llama.cpp loads the GGUF weights straight into this process's memory:

    embeddings   bge-m3, 1024-dim, same model as before -> the SQLCipher schema
                 (vec_chunks embedding FLOAT[1024]) is unchanged
    chat         whatever GGUF you ship, used for both SOAP notes and RAG answers

Two things to know:

  * The models are LAZY. First use pays the load cost (a few seconds). run_app.py
    warms them at launch behind a progress dialog so the clinician never eats
    that mid-session.

  * llama.cpp contexts are NOT thread-safe. One lock guards each model. The chat
    worker and the SOAP-note step can both want the chat model, so the second one
    waits. In practice they never overlap -- you can't chat about a session that
    is still being transcribed -- but the lock means a race can't corrupt a
    context if that assumption ever breaks.
"""
from __future__ import annotations

import threading
from typing import Iterator, Optional

import numpy as np

import config

_embed_model = None
_chat_model = None
_embed_lock = threading.Lock()
_chat_lock = threading.Lock()
_load_lock = threading.Lock()


class ModelMissing(RuntimeError):
    pass


def _require(path) -> str:
    if not path.exists():
        raise ModelMissing(
            f"A model file is missing:\n\n  {path}\n\n"
            "It should have shipped with the app. Reinstall, or see OFFLINE.md "
            "for how to populate models/gguf/."
        )
    return str(path)


def _gpu_layers() -> int:
    # -1 offloads every layer it can. 0 keeps it on the CPU.
    return -1 if config.USE_GPU else 0


# ---------------------------------------------------------------------------
def _load_embedder():
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    with _load_lock:
        if _embed_model is not None:
            return _embed_model
        from llama_cpp import Llama

        _embed_model = Llama(
            model_path=_require(config.EMBED_GGUF),
            embedding=True,
            n_ctx=8192,          # bge-m3's window; chunks are ~600 chars, so ample
            n_gpu_layers=_gpu_layers(),
            verbose=False,
        )
    return _embed_model


def _load_chat():
    global _chat_model
    if _chat_model is not None:
        return _chat_model
    with _load_lock:
        if _chat_model is not None:
            return _chat_model
        from llama_cpp import Llama

        _chat_model = Llama(
            model_path=_require(config.CHAT_GGUF),
            n_ctx=config.CHAT_CTX,
            n_gpu_layers=_gpu_layers(),
            verbose=False,
        )
    return _chat_model


def warm_up(progress=lambda _msg: None) -> None:
    """Called once at launch so the first real request is fast."""
    progress("Loading the embedding model…")
    _load_embedder()
    progress("Loading the language model…")
    _load_chat()
    progress("Ready.")


def unload() -> None:
    global _embed_model, _chat_model
    _embed_model = None
    _chat_model = None


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------
def _pool(raw) -> list[float]:
    """llama.cpp returns either a single pooled vector or one vector per token,
    depending on the pooling type baked into the GGUF. Handle both, so a
    differently-converted bge-m3 file doesn't silently produce garbage that only
    shows up later as bad retrieval."""
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]  # bge-m3 is a CLS-pooled model: first token is the summary
    if arr.ndim != 1:
        raise RuntimeError(f"Unexpected embedding shape {arr.shape}")

    if arr.shape[0] != config.EMBED_DIM:
        raise RuntimeError(
            f"The embedding model returned {arr.shape[0]} dimensions, but the "
            f"vault's vec_chunks table is FLOAT[{config.EMBED_DIM}]. "
            "You've shipped a different embedding model than the vault was built "
            "with — see migrate_reembed.py."
        )

    norm = float(np.linalg.norm(arr))
    return (arr / norm).tolist() if norm > 0 else arr.tolist()


def embed(text: str) -> list[float]:
    """Drop-in replacement for session_store.embed(). Same 1024-dim output."""
    model = _load_embedder()
    with _embed_lock:
        raw = model.embed(text)
    return _pool(raw)


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------
def chat(messages: list[dict], *, temperature: float = 0.2,
         max_tokens: int = 1024) -> str:
    model = _load_chat()
    with _chat_lock:
        out = model.create_chat_completion(
            messages=messages, temperature=temperature, max_tokens=max_tokens,
        )
    return out["choices"][0]["message"]["content"].strip()


def chat_stream(messages: list[dict], *, temperature: float = 0.2,
                max_tokens: int = 1024,
                cancelled=lambda: False) -> Iterator[str]:
    model = _load_chat()
    with _chat_lock:
        for chunk in model.create_chat_completion(
            messages=messages, temperature=temperature,
            max_tokens=max_tokens, stream=True,
        ):
            if cancelled():
                return
            piece = chunk["choices"][0].get("delta", {}).get("content")
            if piece:
                yield piece
