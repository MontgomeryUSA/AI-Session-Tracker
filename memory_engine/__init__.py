"""
memory_engine
=============

A fully local, offline Retrieval-Augmented-Generation (RAG) memory engine
for therapy session transcripts.

Everything the rest of an application needs lives behind the
``MemoryEngine`` facade (see ``memory_engine.engine``). Sub-modules are
intentionally kept small and single-purpose so they can be tested and
swapped independently (e.g. a different embedding backend, a different
vector store).

No network calls are made at runtime except the one-time, optional
download of the embedding model weights from Hugging Face the first
time a given model is used (see ``memory_engine.embeddings`` and the
project README for fully-offline deployment instructions).
"""

from .engine import MemoryEngine
from .config import EngineConfig
from .models import Segment, SearchResult, ContextWindow

__all__ = [
    "MemoryEngine",
    "EngineConfig",
    "Segment",
    "SearchResult",
    "ContextWindow",
]

__version__ = "0.1.0"
