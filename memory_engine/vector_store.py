"""
FAISS vector store.

Vectors are kept entirely separate from SQLite, as required. The
FAISS<->SQLite id mapping is maintained explicitly and simply: we wrap
a flat inner-product index in ``faiss.IndexIDMap`` and always add
vectors using the *SQLite segment id* as the FAISS vector id
(``add_with_ids``). This means there is no separate id-translation
table to keep in sync -- the mapping is the identity, enforced by
construction, which is both simpler and less failure-prone than a
secondary lookup table.

Because vectors are L2-normalized at embedding time, ``IndexFlatIP``
(inner product) search scores are equivalent to cosine similarity in
the range [-1, 1], with 1.0 being a perfect match.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FaissVectorStore:
    """Thin wrapper around a FAISS ``IndexIDMap(IndexFlatIP)``."""

    def __init__(self, dimension: int, index_path: Optional[Path] = None) -> None:
        import faiss  # lazy import: keeps faiss optional for pure-DB use cases

        self._faiss = faiss
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))

        if self.index_path and self.index_path.exists():
            self.load(self.index_path)

    # ------------------------------------------------------------------
    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        if vectors.shape[0] == 0:
            return
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}"
            )
        ids = np.asarray(ids, dtype=np.int64)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.add_with_ids(vectors, ids)

    def remove(self, ids: np.ndarray) -> int:
        """Remove vectors by id. Returns the number actually removed."""
        if len(ids) == 0:
            return 0
        ids = np.asarray(ids, dtype=np.int64)
        return int(self._index.remove_ids(ids))

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for the top_k nearest vectors.

        Returns (scores, ids), both length top_k. ``ids`` contains -1
        for empty slots (e.g. when the index has fewer than top_k
        vectors total).
        """
        if self.count == 0:
            return np.array([]), np.array([], dtype=np.int64)
        query_vector = np.ascontiguousarray(
            query_vector.reshape(1, -1), dtype=np.float32
        )
        top_k = min(top_k, self.count)
        scores, ids = self._index.search(query_vector, top_k)
        return scores[0], ids[0]

    @property
    def count(self) -> int:
        return int(self._index.ntotal)

    def reset(self) -> None:
        self._index = self._faiss.IndexIDMap(self._faiss.IndexFlatIP(self.dimension))

    # ------------------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> None:
        target = Path(path) if path else self.index_path
        if target is None:
            raise ValueError("No index_path configured and none provided to save()")
        target.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(target))

        meta_path = target.with_suffix(target.suffix + ".meta.json")
        meta_path.write_text(
            json.dumps({"dimension": self.dimension, "count": self.count})
        )
        logger.info("Saved FAISS index (%d vectors) to %s", self.count, target)

    def load(self, path: Path) -> None:
        path = Path(path)
        self._index = self._faiss.read_index(str(path))
        if self._index.d != self.dimension:
            raise ValueError(
                f"Loaded index dimension {self._index.d} does not match "
                f"configured embedder dimension {self.dimension}. "
                "This usually means the embedding model changed -- run "
                "MemoryEngine.rebuild_index()."
            )
