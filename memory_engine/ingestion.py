"""
Transcript ingestion.

Phase 1: recursively scan a directory, load every transcript JSON file,
iterate through its segments, and create one Segment record per segment.

Phase 5 (forward compatibility): if a file lives under a
``Patients/<patient_id>/sessions/<file>.json`` style path, ``patient_id``
is automatically extracted. ``session_id`` defaults to the filename
stem, which today doubles as ``conversation_id`` per the current flat
layout, but is kept as a distinct column so a future migration (e.g.
namespacing conversation_id by patient to avoid filename collisions
across patients) only has to backfill data, not change the schema.

The transcript JSON format itself is treated as immutable, untouched
source data -- this module only *reads* it.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .database import Database
from .models import Segment

logger = logging.getLogger(__name__)


@dataclass
class FileIngestResult:
    """Outcome of ingesting a single transcript file."""

    file_path: str
    status: str  # "new" | "updated" | "unchanged" | "error"
    removed_segment_ids: list[int] = field(default_factory=list)
    inserted_segments: list[Segment] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class IngestionStats:
    """Aggregate summary returned to the caller of ``ingest_directory``."""

    files_scanned: int = 0
    files_new: int = 0
    files_updated: int = 0
    files_unchanged: int = 0
    files_failed: int = 0
    segments_inserted: int = 0
    segments_removed: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "files_scanned": self.files_scanned,
            "files_new": self.files_new,
            "files_updated": self.files_updated,
            "files_unchanged": self.files_unchanged,
            "files_failed": self.files_failed,
            "segments_inserted": self.segments_inserted,
            "segments_removed": self.segments_removed,
            "errors": self.errors,
        }


def _content_hash(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def _derive_patient_id(file_path: Path) -> Optional[str]:
    """Best-effort detection of the Phase 5 ``Patients/<id>/sessions/*`` layout.

    Returns None for the current flat layout, which is the expected
    case for today's sample data.
    """
    parts = file_path.parts
    for i, part in enumerate(parts):
        if part.lower() == "patients" and i + 1 < len(parts):
            return parts[i + 1]
    return None


class TranscriptIngestor:
    """Reads transcript JSON files and writes Segment records to SQLite.

    Kept independent of the embedding/vector store layers: this class
    only knows about files and SQLite rows. ``MemoryEngine`` is
    responsible for turning newly-inserted segments into embeddings.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------
    def scan(self, root: Path) -> list[Path]:
        """Recursively find transcript JSON files under ``root``."""
        if not root.exists():
            raise FileNotFoundError(f"Directory does not exist: {root}")
        return sorted(p for p in root.rglob("*.json") if p.is_file())

    # ------------------------------------------------------------------
    def parse_file(self, file_path: Path, raw_bytes: bytes) -> list[Segment]:
        """Parse one transcript file into a list of un-persisted Segments."""
        data = json.loads(raw_bytes.decode("utf-8"))
        segments_raw = data.get("segments")
        if not isinstance(segments_raw, list):
            raise ValueError("Transcript JSON is missing a 'segments' list")

        conversation_id = file_path.stem
        session_id = conversation_id
        patient_id = _derive_patient_id(file_path)
        created_at = datetime.now(timezone.utc).isoformat()

        segments: list[Segment] = []
        for idx, raw in enumerate(segments_raw):
            if not isinstance(raw, dict):
                logger.warning(
                    "Skipping malformed segment (index %d) in %s: not an object",
                    idx,
                    file_path,
                )
                continue
            text = str(raw.get("text", "")).strip()
            if not text:
                # Empty utterances carry no retrievable signal; skip them
                # but keep segment_index aligned with the original file
                # by not consuming an index for skipped entries -- context
                # windows only need a stable order among *stored* rows.
                continue
            speaker = str(raw.get("speaker", "UNKNOWN"))
            try:
                start_time = float(raw.get("start", 0.0))
                end_time = float(raw.get("end", start_time))
            except (TypeError, ValueError):
                logger.warning(
                    "Non-numeric start/end in segment %d of %s; defaulting to 0.0",
                    idx,
                    file_path,
                )
                start_time, end_time = 0.0, 0.0

            segments.append(
                Segment(
                    conversation_id=conversation_id,
                    session_id=session_id,
                    patient_id=patient_id,
                    file_path=str(file_path),
                    segment_index=len(segments),
                    speaker=speaker,
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                    created_at=created_at,
                )
            )
        return segments

    # ------------------------------------------------------------------
    def ingest_directory(self, root: Path) -> tuple[list[FileIngestResult], IngestionStats]:
        """Ingest every transcript file under ``root``.

        Idempotent: unchanged files (matched by content hash) are
        skipped; changed files have their old segments removed and
        replaced. Returns per-file results (so the caller can update a
        vector index incrementally) plus an aggregate summary.
        """
        stats = IngestionStats()
        results: list[FileIngestResult] = []

        files = self.scan(root)
        stats.files_scanned = len(files)

        for file_path in files:
            str_path = str(file_path)
            try:
                raw_bytes = file_path.read_bytes()
                content_hash = _content_hash(raw_bytes)
                existing = self.db.get_source_file(str_path)

                if existing is not None and existing["content_hash"] == content_hash:
                    stats.files_unchanged += 1
                    results.append(FileIngestResult(file_path=str_path, status="unchanged"))
                    continue

                segments = self.parse_file(file_path, raw_bytes)

                removed_ids: list[int] = []
                if existing is not None:
                    removed_ids = self.db.delete_segments_by_file(str_path)
                    stats.files_updated += 1
                    status = "updated"
                else:
                    stats.files_new += 1
                    status = "new"

                inserted_ids = self.db.insert_segments(segments)
                for seg, seg_id in zip(segments, inserted_ids):
                    seg.id = seg_id

                conversation_id = file_path.stem
                patient_id = _derive_patient_id(file_path)
                self.db.upsert_source_file(
                    file_path=str_path,
                    content_hash=content_hash,
                    conversation_id=conversation_id,
                    patient_id=patient_id,
                    session_id=conversation_id,
                    segment_count=len(segments),
                )

                stats.segments_inserted += len(segments)
                stats.segments_removed += len(removed_ids)

                results.append(
                    FileIngestResult(
                        file_path=str_path,
                        status=status,
                        removed_segment_ids=removed_ids,
                        inserted_segments=segments,
                    )
                )
                logger.info(
                    "%s: %s (%d segments)", status, file_path.name, len(segments)
                )

            except (json.JSONDecodeError, ValueError, OSError) as exc:
                stats.files_failed += 1
                msg = f"{file_path}: {exc}"
                stats.errors.append(msg)
                results.append(
                    FileIngestResult(file_path=str_path, status="error", error=str(exc))
                )
                logger.error("Failed to ingest %s: %s", file_path, exc)

        return results, stats
