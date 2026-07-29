"""
workers.py
==========
Two background jobs, because both of them would freeze the window if they ran on
the UI thread.

  IngestWorker  minutes long (WhisperX is the slow part). Emits stage changes and
                log lines as it goes.
  ChatWorker    seconds long, but streams, and the whole point is watching it
                arrive.

Each worker makes its own database connection via vault.con(), because SQLCipher
connections belong to the thread that created them.

Qt signals are the thread boundary: everything crossing back to the UI goes
through one, which is the only safe way to touch widgets from another thread.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

import pipeline
import rag_chat
import vault


class IngestWorker(QThread):
    stage = Signal(str)          # preparing | transcribing | summarizing | indexing | done
    log = Signal(str)
    finished_ok = Signal(object)  # pipeline.SessionResult
    failed = Signal(str)

    def __init__(self, audio: Path, passphrase: str, patient_id: int,
                 alias: str, role_map: dict, parent=None):
        super().__init__(parent)
        self._audio = audio
        self._passphrase = passphrase
        self._patient_id = patient_id
        self._alias = alias
        self._role_map = role_map
        self._cancel = False

    def cancel(self) -> None:
        """Cooperative: checked between stages. WhisperX itself is a subprocess
        we don't interrupt mid-flight, so a cancel during transcription takes
        effect when that stage ends rather than instantly."""
        self._cancel = True

    def run(self) -> None:
        try:
            result = pipeline.process_audio(
                self._audio,
                self._passphrase,
                patient_id=self._patient_id,
                patient_alias=self._alias,
                role_map=self._role_map,
                log=self.log.emit,
                stage=self.stage.emit,
                cancelled=lambda: self._cancel,
            )
            self.finished_ok.emit(result)
        except InterruptedError:
            self.failed.emit("Cancelled.")
        except Exception as exc:
            self.failed.emit(str(exc) or exc.__class__.__name__)


class ChatWorker(QThread):
    sources = Signal(object)   # {"sources": [...], "search_query": str}
    token = Signal(str)
    finished_ok = Signal(str)  # the complete answer
    failed = Signal(str)

    def __init__(self, patient_id: int, question: str, history: list[dict], parent=None):
        super().__init__(parent)
        self._patient_id = patient_id
        self._question = question
        self._history = list(history)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        answer: list[str] = []
        try:
            for kind, payload in rag_chat.turn(
                vault.con(), self._patient_id, self._question, self._history
            ):
                if self._cancel:
                    break
                if kind == "sources":
                    self.sources.emit(payload)
                else:
                    answer.append(payload)
                    self.token.emit(payload)
            self.finished_ok.emit("".join(answer))
        except Exception as exc:
            self.failed.emit(str(exc) or exc.__class__.__name__)
