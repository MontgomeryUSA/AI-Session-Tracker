"""
vault.py
========
One place that knows the passphrase, and one connection per thread.

SQLCipher connections are bound to the thread that made them, and the app has
three threads that touch the database (the UI, the transcription worker, the
chat worker). So: hold the passphrase in memory once, hand out a lazily-created
connection per thread.

The passphrase is never written anywhere. Close the app and it's gone. That is
the point of the vault -- an attacker with the .db file has ciphertext.
"""
from __future__ import annotations

import threading

import config
import session_store as store

_local = threading.local()
_passphrase: str | None = None
_lock = threading.Lock()


class Locked(RuntimeError):
    pass


def unlock(passphrase: str) -> None:
    """Open the vault, or raise if the passphrase is wrong.

    SQLCipher doesn't reject a bad key at PRAGMA time -- it fails on the first
    real read, because a wrong key just means the file decrypts to garbage that
    doesn't look like SQLite. So we force a read here and let it fail now,
    rather than three screens later.
    """
    global _passphrase
    con = store.connect(str(config.DB_PATH), passphrase)
    try:
        store.init_schema(con)                       # creates the file if new
        con.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except Exception as exc:
        con.close()
        raise Locked("That passphrase doesn't open this vault.") from exc

    with _lock:
        _passphrase = passphrase
    _local.con = con


def is_unlocked() -> bool:
    return _passphrase is not None


def con():
    """The calling thread's connection. Creates it on first use."""
    if _passphrase is None:
        raise Locked("The vault is locked.")
    existing = getattr(_local, "con", None)
    if existing is None:
        existing = store.connect(str(config.DB_PATH), _passphrase)
        store.init_schema(existing)
        _local.con = existing
    return existing


def lock() -> None:
    global _passphrase
    with _lock:
        _passphrase = None
    existing = getattr(_local, "con", None)
    if existing is not None:
        existing.close()
        _local.con = None


# ---------------------------------------------------------------------------
# reads the UI needs
# ---------------------------------------------------------------------------
def patients() -> list[dict]:
    rows = con().execute("SELECT id, alias FROM patient ORDER BY id").fetchall()
    return [{"id": r[0], "alias": r[1]} for r in rows]


def sessions(patient_id: int) -> list[dict]:
    rows = con().execute(
        """
        SELECT s.id, s.ts, s.note, s.audio_path,
               (SELECT COUNT(*)     FROM chunk c WHERE c.session_id = s.id),
               (SELECT MAX(c."end") FROM chunk c WHERE c.session_id = s.id)
        FROM session s
        WHERE s.patient_id = ?
        ORDER BY s.id DESC
        """,
        (patient_id,),
    ).fetchall()
    return [
        {"id": r[0], "ts": r[1], "note": r[2] or "", "audio_path": r[3],
         "chunks": r[4] or 0, "duration": float(r[5] or 0.0)}
        for r in rows
    ]


def session(session_id: int) -> dict | None:
    r = con().execute(
        "SELECT id, patient_id, ts, transcript, note, audio_path FROM session WHERE id = ?",
        (session_id,),
    ).fetchone()
    if not r:
        return None
    dur = con().execute(
        'SELECT MAX("end") FROM chunk WHERE session_id = ?', (session_id,)
    ).fetchone()[0]
    return {"id": r[0], "patient_id": r[1], "ts": r[2], "transcript": r[3],
            "note": r[4], "audio_path": r[5], "duration": float(dur or 0.0)}
