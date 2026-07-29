"""
Core data structures shared across the engine.

Kept dependency-free (no sqlite3 / faiss imports here) so they can be
used freely by any layer, including future UI code, without dragging
in storage internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Segment:
    """A single utterance/turn from a transcript file.

    ``id`` is None for segments that have not yet been persisted to
    SQLite (e.g. while parsing a file, before insertion).

    ``patient_id`` and ``session_id`` are forward-looking fields for the
    Phase 5 ``Patients/<id>/sessions/<file>.json`` layout. They are
    populated on a best-effort basis today and are simply absent
    (``None`` / equal to ``conversation_id``) for the current flat
    transcript layout.
    """

    conversation_id: str
    session_id: str
    file_path: str
    segment_index: int
    speaker: str
    start_time: float
    end_time: float
    text: str
    created_at: str
    patient_id: Optional[str] = None
    id: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "patient_id": self.patient_id,
            "file_path": self.file_path,
            "segment_index": self.segment_index,
            "speaker": self.speaker,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "created_at": self.created_at,
        }


@dataclass
class SearchResult:
    """A single semantic search hit, ready for display or downstream LLM use."""

    score: float
    segment_id: int
    conversation_id: str
    session_id: str
    speaker: str
    start_time: float
    end_time: float
    text: str
    patient_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "segment_id": self.segment_id,
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "patient_id": self.patient_id,
            "speaker": self.speaker,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
        }


@dataclass
class ContextWindow:
    """A reconstructed slice of conversation surrounding a matched segment."""

    matched_segment_id: int
    score: float
    conversation_id: str
    session_id: str
    segments: list[Segment] = field(default_factory=list)

    def formatted(self) -> str:
        """Render the window as ``Speaker: text`` lines, in chronological order.

        The matched segment is marked with a ``>>`` prefix so downstream
        consumers (CLI, future LLM prompts) can easily highlight it.
        """
        lines = []
        for seg in self.segments:
            marker = ">> " if seg.id == self.matched_segment_id else "   "
            lines.append(f"{marker}{seg.speaker}: {seg.text}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched_segment_id": self.matched_segment_id,
            "score": self.score,
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "segments": [s.to_dict() for s in self.segments],
            "formatted_text": self.formatted(),
        }
