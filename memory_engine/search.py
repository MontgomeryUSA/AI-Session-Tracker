"""
Semantic search (Phase 3) and conversation-window context retrieval (Phase 4).

Kept as a standalone service (rather than folded directly into
``MemoryEngine``) so it only needs a Database, a vector store and an
Embedder -- easy to unit test with fakes, and easy to reuse if a future
batch/offline scoring job wants search without the full engine.
"""
from __future__ import annotations

from .database import Database
from .embeddings import Embedder
from .models import ContextWindow, SearchResult
from .vector_store import FaissVectorStore


class SearchService:
    def __init__(self, db: Database, vector_store: FaissVectorStore, embedder: Embedder) -> None:
        self.db = db
        self.vector_store = vector_store
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Phase 3
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Embed ``query``, search FAISS, and hydrate full records from SQLite."""
        if not query or not query.strip():
            return []

        query_vector = self.embedder.encode_query(query)
        scores, ids = self.vector_store.search(query_vector, top_k)

        results: list[SearchResult] = []
        for score, seg_id in zip(scores, ids):
            if seg_id == -1:
                continue
            segment = self.db.get_segment(int(seg_id))
            if segment is None:
                # Vector exists but the underlying row was deleted
                # without a corresponding index update -- skip rather
                # than crash; rebuild_index() will fix this.
                continue
            results.append(
                SearchResult(
                    score=float(score),
                    segment_id=segment.id,  # type: ignore[arg-type]
                    conversation_id=segment.conversation_id,
                    session_id=segment.session_id,
                    patient_id=segment.patient_id,
                    speaker=segment.speaker,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    text=segment.text,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Phase 4
    # ------------------------------------------------------------------
    def retrieve_context(
        self,
        query: str,
        top_k: int = 10,
        before: int = 2,
        after: int = 2,
    ) -> list[ContextWindow]:
        """Like ``search``, but each hit is expanded into a surrounding
        conversation window (``before`` segments earlier, ``after`` later,
        from the same source file) instead of being returned in isolation.
        """
        matches = self.search(query, top_k)
        windows: list[ContextWindow] = []
        for match in matches:
            segment = self.db.get_segment(match.segment_id)
            if segment is None:
                continue
            context_segments = self.db.get_context(
                segment.file_path, segment.segment_index, before, after
            )
            windows.append(
                ContextWindow(
                    matched_segment_id=match.segment_id,
                    score=match.score,
                    conversation_id=match.conversation_id,
                    session_id=match.session_id,
                    segments=context_segments,
                )
            )
        return windows
