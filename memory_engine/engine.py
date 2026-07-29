"""
Phase 6: the MemoryEngine service-layer facade.

This is the ONLY class the rest of the application should depend on.
Everything else in this package (Database, TranscriptIngestor,
Embedder, FaissVectorStore, SearchService) is an internal collaborator
that ``MemoryEngine`` wires together. This keeps the integration
surface small and stable even as internals evolve (different vector
store, different embedding model, sharded per-patient indices, etc).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import EngineConfig
from .database import Database
from .embeddings import Embedder, SentenceTransformerEmbedder
from .ingestion import IngestionStats, TranscriptIngestor
from .search import SearchService
from .vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


class MemoryEngine:
    """Local RAG memory engine for therapy session transcripts.

    Example:
        >>> engine = MemoryEngine()
        >>> engine.ingest_directory("./transcripts")
        >>> results = engine.search("sleep problems", top_k=5)
        >>> windows = engine.retrieve_context("sleep problems", top_k=5)
        >>> engine.close()
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.config.ensure_data_dir()

        self.db = Database(self.config.db_path)
        self.db.initialize_schema()

        # Allow dependency injection of a fake/alternative embedder
        # (used heavily in tests, and useful if a desktop app wants to
        # share one already-loaded model instance across components).
        self.embedder: Embedder = embedder or SentenceTransformerEmbedder(
            model_name=self.config.model_name,
            cache_folder=self.config.model_cache_dir,
            local_files_only=self.config.local_files_only,
            query_instruction=self.config.query_instruction,
        )

        self.vector_store = FaissVectorStore(
            dimension=self.embedder.dimension,
            index_path=self.config.index_path,
        )

        self.ingestor = TranscriptIngestor(self.db)
        self.search_service = SearchService(self.db, self.vector_store, self.embedder)

        self._warn_if_index_out_of_sync()

    # ------------------------------------------------------------------
    def _warn_if_index_out_of_sync(self) -> None:
        db_count = self.db.count_segments()
        if db_count > 0 and self.vector_store.count == 0:
            logger.warning(
                "Database has %d segments but the vector index is empty. "
                "Call rebuild_index() to (re)generate embeddings.",
                db_count,
            )
        elif db_count > 0 and self.vector_store.count != db_count:
            logger.warning(
                "Vector index has %d vectors but database has %d segments. "
                "They may be out of sync; consider calling rebuild_index().",
                self.vector_store.count,
                db_count,
            )

    # ------------------------------------------------------------------
    # Phase 1/2 combined: ingest files, embed, index
    # ------------------------------------------------------------------
    def ingest_directory(self, path: str | Path) -> dict[str, Any]:
        """Recursively ingest every transcript JSON file under ``path``.

        Safe to call repeatedly on the same directory (and on a growing
        directory of new files): already-ingested, unchanged files are
        skipped; changed files are fully re-indexed.
        """
        root = Path(path)
        results, stats = self.ingestor.ingest_directory(root)

        ids_to_remove: list[int] = []
        new_texts: list[str] = []
        new_ids: list[int] = []

        for result in results:
            ids_to_remove.extend(result.removed_segment_ids)
            for seg in result.inserted_segments:
                assert seg.id is not None
                new_texts.append(seg.text)
                new_ids.append(seg.id)

        if ids_to_remove:
            removed = self.vector_store.remove(np.array(ids_to_remove, dtype=np.int64))
            logger.info("Removed %d stale vectors", removed)

        if new_texts:
            vectors = self.embedder.encode_passages(new_texts)
            self.vector_store.add(np.array(new_ids, dtype=np.int64), vectors)

        if ids_to_remove or new_texts:
            self.vector_store.save(self.config.index_path)

        return stats.to_dict()

    # ------------------------------------------------------------------
    # Phase 3
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.search_service.search(query, top_k)]

    # ------------------------------------------------------------------
    # Phase 4
    # ------------------------------------------------------------------
    def retrieve_context(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        windows = self.search_service.retrieve_context(
            query,
            top_k,
            before=self.config.context_before,
            after=self.config.context_after,
        )
        return [w.to_dict() for w in windows]

    # ------------------------------------------------------------------
    # Phase 6 helper
    # ------------------------------------------------------------------
    def get_session(self, session_id: str) -> dict[str, Any]:
        """Return the full reconstructed transcript for one session."""
        segments = self.db.get_session(session_id)
        return {
            "session_id": session_id,
            "segment_count": len(segments),
            "segments": [s.to_dict() for s in segments],
            "transcript": "\n".join(f"{s.speaker}: {s.text}" for s in segments),
        }

    # ------------------------------------------------------------------
    def rebuild_index(self) -> int:
        """Recompute embeddings for every segment in SQLite and rebuild
        the FAISS index from scratch.

        Use this after changing the embedding model, after recovering
        from a deleted/corrupted index file, or any time
        ``_warn_if_index_out_of_sync`` flags a mismatch.
        """
        all_segments = self.db.get_all_segments()
        self.vector_store.reset()

        if all_segments:
            texts = [s.text for s in all_segments]
            ids = np.array([s.id for s in all_segments], dtype=np.int64)
            vectors = self.embedder.encode_passages(texts)
            self.vector_store.add(ids, vectors)

        self.vector_store.save(self.config.index_path)
        logger.info("Rebuilt index with %d vectors", len(all_segments))
        return len(all_segments)

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "MemoryEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
