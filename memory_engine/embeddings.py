"""
Embedding pipeline.

Phase 2 requirement: use sentence-transformers / BAAI/bge-small-en-v1.5
locally, no API calls. ``SentenceTransformerEmbedder`` below does exactly
that.

``Embedder`` is defined as a ``Protocol`` so the rest of the codebase
(and tests) depend only on the interface, not on sentence-transformers
itself. This keeps unit tests fast (no model load) and makes it easy to
later swap in a different local model without touching ``MemoryEngine``.

A note on "fully offline": running inference with a sentence-transformers
model is 100% local -- no network call happens during ``encode()``.
The *first* time a given model name is used on a machine, however,
sentence-transformers/huggingface_hub will attempt to download the model
weights from Hugging Face if they are not already cached. For a true
air-gapped / HIPAA-conscious deployment:

1. On a machine with internet access, pre-download the model once
   (e.g. by running this module's loader, or
   ``huggingface-cli download BAAI/bge-small-en-v1.5``).
2. Copy the resulting cache folder (or pass ``model_cache_dir``) onto
   the target laptop.
3. Construct ``SentenceTransformerEmbedder`` with ``local_files_only=True``
   so it never attempts a network call, failing loudly instead if the
   weights are missing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .config import DEFAULT_QUERY_INSTRUCTION


@runtime_checkable
class Embedder(Protocol):
    """Interface the rest of the engine relies on."""

    dimension: int

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of transcript segment texts -> (n, dim) float32 array."""
        ...

    def encode_query(self, text: str) -> np.ndarray:
        """Embed a single search query -> (dim,) float32 array."""
        ...


class SentenceTransformerEmbedder:
    """Local embedding backend using sentence-transformers.

    Defaults to BAAI/bge-small-en-v1.5 per the Phase 2 requirement.
    All vectors are L2-normalized so that FAISS inner-product search
    (``IndexFlatIP``) is equivalent to cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
        cache_folder: Optional[Path] = None,
        local_files_only: bool = False,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        batch_size: int = 32,
    ) -> None:
        # Imported lazily so importing this *module* never requires
        # sentence-transformers/torch unless this class is actually
        # instantiated (keeps lightweight test runs fast).
        from sentence_transformers import SentenceTransformer

        if local_files_only:
            # Belt-and-suspenders: also set the HF env var so any
            # transitive huggingface_hub call respects offline mode even
            # if a future sentence-transformers version adds one we
            # don't pass the kwarg through to.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self._model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=str(cache_folder) if cache_folder else None,
            local_files_only=local_files_only,
        )
        self.dimension: int = self._model.get_sentence_embedding_dimension()
        self.query_instruction = query_instruction
        self.batch_size = batch_size

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        embeddings = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        prefixed = f"{self.query_instruction}{text}"
        embedding = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return embedding.astype(np.float32)
