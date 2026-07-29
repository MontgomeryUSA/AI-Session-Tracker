"""
SQLite persistence layer.

Design notes
------------
- A single ``transcript_segments`` table stores one row per transcript
  segment (Phase 1), already carrying the forward-looking ``patient_id``
  and ``session_id`` columns required by Phase 5 so no migration is
  needed later -- only a backfill of those two columns once the real
  ``Patients/<id>/sessions/<file>.json`` layout exists.
- A second ``source_files`` table tracks which files have already been
  ingested (by content hash) so ``ingest_directory`` is idempotent and
  safe to re-run on a directory that already contains indexed files,
  and so changed files can be cleanly re-indexed (old segments + their
  embeddings removed, new ones added).
- No transcript text is ever written to logs -- only counts, ids and
  file paths -- since this data is PHI in a real deployment.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import Segment

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcript_segments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT    NOT NULL,
    patient_id      TEXT,
    session_id      TEXT    NOT NULL,
    file_path       TEXT    NOT NULL,
    segment_index   INTEGER NOT NULL,
    speaker         TEXT    NOT NULL,
    start_time      REAL    NOT NULL,
    end_time        REAL    NOT NULL,
    text            TEXT    NOT NULL,
    created_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_segments_file_path
    ON transcript_segments (file_path, segment_index);

CREATE INDEX IF NOT EXISTS idx_segments_conversation
    ON transcript_segments (conversation_id);

CREATE INDEX IF NOT EXISTS idx_segments_session
    ON transcript_segments (session_id);

CREATE INDEX IF NOT EXISTS idx_segments_patient
    ON transcript_segments (patient_id);

CREATE TABLE IF NOT EXISTS source_files (
    file_path      TEXT PRIMARY KEY,
    content_hash   TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    patient_id     TEXT,
    session_id     TEXT NOT NULL,
    segment_count  INTEGER NOT NULL,
    ingested_at    TEXT NOT NULL
);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_segment(row: sqlite3.Row) -> Segment:
    return Segment(
        id=row["id"],
        conversation_id=row["conversation_id"],
        patient_id=row["patient_id"],
        session_id=row["session_id"],
        file_path=row["file_path"],
        segment_index=row["segment_index"],
        speaker=row["speaker"],
        start_time=row["start_time"],
        end_time=row["end_time"],
        text=row["text"],
        created_at=row["created_at"],
    )


class Database:
    """Thin, explicit wrapper around the sqlite3 connection.

    Intended for single-process desktop use. Not designed for concurrent
    multi-process writers; SQLite's WAL mode is enabled so concurrent
    *readers* alongside a single writer (e.g. a future UI process) work
    fine.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def initialize_schema(self) -> None:
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # source_files bookkeeping (for idempotent / incremental ingestion)
    # ------------------------------------------------------------------
    def get_source_file(self, file_path: str) -> Optional[sqlite3.Row]:
        cur = self._conn.execute(
            "SELECT * FROM source_files WHERE file_path = ?", (file_path,)
        )
        return cur.fetchone()

    def upsert_source_file(
        self,
        file_path: str,
        content_hash: str,
        conversation_id: str,
        patient_id: Optional[str],
        session_id: str,
        segment_count: int,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO source_files
                (file_path, content_hash, conversation_id, patient_id, session_id, segment_count, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                content_hash=excluded.content_hash,
                conversation_id=excluded.conversation_id,
                patient_id=excluded.patient_id,
                session_id=excluded.session_id,
                segment_count=excluded.segment_count,
                ingested_at=excluded.ingested_at
            """,
            (file_path, content_hash, conversation_id, patient_id, session_id, segment_count, _utcnow()),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # transcript_segments writes
    # ------------------------------------------------------------------
    def delete_segments_by_file(self, file_path: str) -> list[int]:
        """Delete all segments for a file, returning their (now-stale) ids
        so the caller can also remove the matching vectors from FAISS."""
        cur = self._conn.execute(
            "SELECT id FROM transcript_segments WHERE file_path = ?", (file_path,)
        )
        ids = [row["id"] for row in cur.fetchall()]
        if ids:
            self._conn.execute(
                "DELETE FROM transcript_segments WHERE file_path = ?", (file_path,)
            )
            self._conn.commit()
        return ids

    def insert_segments(self, segments: list[Segment]) -> list[int]:
        """Insert segments (which must not yet have an ``id``) and return
        the assigned ids, in the same order as the input list."""
        ids: list[int] = []
        for seg in segments:
            cur = self._conn.execute(
                """
                INSERT INTO transcript_segments
                    (conversation_id, patient_id, session_id, file_path, segment_index,
                     speaker, start_time, end_time, text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    seg.conversation_id,
                    seg.patient_id,
                    seg.session_id,
                    seg.file_path,
                    seg.segment_index,
                    seg.speaker,
                    seg.start_time,
                    seg.end_time,
                    seg.text,
                    seg.created_at,
                ),
            )
            ids.append(cur.lastrowid)
        self._conn.commit()
        return ids

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get_segment(self, segment_id: int) -> Optional[Segment]:
        cur = self._conn.execute(
            "SELECT * FROM transcript_segments WHERE id = ?", (segment_id,)
        )
        row = cur.fetchone()
        return _row_to_segment(row) if row else None

    def get_segments_by_file(self, file_path: str) -> list[Segment]:
        cur = self._conn.execute(
            "SELECT * FROM transcript_segments WHERE file_path = ? ORDER BY segment_index",
            (file_path,),
        )
        return [_row_to_segment(r) for r in cur.fetchall()]

    def get_context(
        self, file_path: str, segment_index: int, before: int, after: int
    ) -> list[Segment]:
        cur = self._conn.execute(
            """
            SELECT * FROM transcript_segments
            WHERE file_path = ? AND segment_index BETWEEN ? AND ?
            ORDER BY segment_index
            """,
            (file_path, segment_index - before, segment_index + after),
        )
        return [_row_to_segment(r) for r in cur.fetchall()]

    def get_session(self, session_id: str) -> list[Segment]:
        cur = self._conn.execute(
            "SELECT * FROM transcript_segments WHERE session_id = ? ORDER BY segment_index",
            (session_id,),
        )
        return [_row_to_segment(r) for r in cur.fetchall()]

    def get_all_segments(self) -> list[Segment]:
        cur = self._conn.execute(
            "SELECT * FROM transcript_segments ORDER BY file_path, segment_index"
        )
        return [_row_to_segment(r) for r in cur.fetchall()]

    def count_segments(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) AS c FROM transcript_segments")
        return int(cur.fetchone()["c"])

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
