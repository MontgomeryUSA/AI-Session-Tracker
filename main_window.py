"""
main_window.py
==============
The window. Left rail captures, right side interrogates.

Both halves talk to the same two functions the terminal tools always used
(pipeline.process_audio, rag_chat.turn) -- the GUI is a front door, not a
rewrite. main.py and chat_session.py still work.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

import config
import theme
import vault
from recorder import Recorder
from widgets import AudioPlayer, LevelMeter, RecordButton, Ribbon, StepRail, mmss
from workers import ChatWorker, IngestWorker

SEEDS = [
    "What was said about sleep?",
    "Summarise the last session.",
    "Has the patient mentioned family?",
]


def _eyebrow(text: str) -> QLabel:
    lab = QLabel(text.upper())
    lab.setObjectName("Eyebrow")
    return lab


class MainWindow(QMainWindow):
    level_captured = Signal(float)   # audio thread -> UI thread, safely

    def __init__(self, passphrase: str):
        super().__init__()
        self._passphrase = passphrase
        self._history: list[dict] = []
        self._sessions: list[dict] = []
        self._ingest: IngestWorker | None = None
        self._chat: ChatWorker | None = None

        self.setWindowTitle(config.APP_NAME)
        self.resize(1180, 840)
        self.setMinimumSize(900, 620)

        self.player = AudioPlayer(self)
        self.recorder = Recorder(on_level=self.level_captured.emit)
        self.level_captured.connect(self._on_level, Qt.QueuedConnection)

        root = QWidget()
        row = QHBoxLayout(root)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self._build_rail())
        row.addWidget(self._build_main(), 1)
        self.setCentralWidget(root)

        self._pulse = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        QShortcut(QKeySequence("Ctrl+R"), self, self._toggle_record)
        QShortcut(QKeySequence("Ctrl+L"), self, lambda: self.ask.setFocus())

        self._reload_sessions()

    # ======================================================================
    # left rail
    # ======================================================================
    def _build_rail(self) -> QWidget:
        rail = QFrame()
        rail.setObjectName("Rail")
        rail.setFixedWidth(390)
        col = QVBoxLayout(rail)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        # -- brand
        brand = QFrame()
        brand.setObjectName("Brand")
        b = QVBoxLayout(brand)
        b.setContentsMargins(22, 20, 22, 16)
        b.setSpacing(4)
        name = QLabel(config.APP_NAME)
        name.setObjectName("BrandName")
        sub = QLabel("LOCAL · ENCRYPTED · OFFLINE")
        sub.setObjectName("BrandSub")
        b.addWidget(name)
        b.addWidget(sub)
        col.addWidget(brand)

        # -- capture
        cap = QFrame()
        cap.setObjectName("Capture")
        c = QVBoxLayout(cap)
        c.setContentsMargins(22, 20, 22, 20)
        c.setSpacing(14)

        who = QGridLayout()
        who.setHorizontalSpacing(8)
        who.setVerticalSpacing(4)
        self.pid = QSpinBox()
        self.pid.setRange(1, 999999)
        self.pid.setValue(1)
        self.pid.valueChanged.connect(self._on_patient_changed)
        self.alias = QLineEdit("PT-0001")
        who.addWidget(_eyebrow("Patient ID"), 0, 0)
        who.addWidget(_eyebrow("Alias"), 0, 1)
        who.addWidget(self.pid, 1, 0)
        who.addWidget(self.alias, 1, 1)
        c.addLayout(who)

        rec_row = QHBoxLayout()
        rec_row.setSpacing(12)
        self.rec = RecordButton()
        self.rec.clicked.connect(self._toggle_record)
        self.meter = LevelMeter()
        rec_row.addWidget(self.rec)
        rec_row.addWidget(self.meter, 1)
        c.addLayout(rec_row)

        state = QHBoxLayout()
        self.clock = QLabel("00:00")
        self.clock.setStyleSheet(
            f'font-family:"{theme.DATA_FAMILY}";font-size:18px;color:{theme.INK};'
        )
        self.pick = QPushButton("or open an audio file")
        self.pick.setObjectName("Link")
        self.pick.setCursor(Qt.PointingHandCursor)
        self.pick.clicked.connect(self._open_file)
        state.addWidget(self.clock)
        state.addStretch(1)
        state.addWidget(self.pick)
        c.addLayout(state)
        col.addWidget(cap)

        # -- progress
        self.work = QFrame()
        self.work.setObjectName("Work")
        w = QVBoxLayout(self.work)
        w.setContentsMargins(22, 18, 22, 18)
        w.setSpacing(12)
        w.addWidget(_eyebrow("Processing"))
        self.rail = StepRail()
        w.addWidget(self.rail)
        self.log = QPlainTextEdit()
        self.log.setObjectName("Log")
        self.log.setReadOnly(True)
        self.log.setFixedHeight(104)
        w.addWidget(self.log)
        self.work.hide()
        col.addWidget(self.work)

        # -- sessions
        sess = QWidget()
        s = QVBoxLayout(sess)
        s.setContentsMargins(22, 18, 22, 10)
        s.setSpacing(10)
        s.addWidget(_eyebrow("Sessions on file"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        self.slist = QVBoxLayout(holder)
        self.slist.setContentsMargins(0, 0, 6, 0)
        self.slist.setSpacing(7)
        self.slist.addStretch(1)
        scroll.setWidget(holder)
        s.addWidget(scroll, 1)
        col.addWidget(sess, 1)

        return rail

    # ======================================================================
    # right side
    # ======================================================================
    def _build_main(self) -> QWidget:
        main = QWidget()
        col = QVBoxLayout(main)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        self.banner = QLabel()
        self.banner.setObjectName("Banner")
        self.banner.setWordWrap(True)
        self.banner.hide()
        col.addWidget(self.banner)

        self.thread_scroll = QScrollArea()
        self.thread_scroll.setWidgetResizable(True)
        holder = QWidget()
        wrap = QHBoxLayout(holder)
        wrap.setContentsMargins(40, 36, 40, 24)
        inner = QWidget()
        inner.setMaximumWidth(700)
        self.thread = QVBoxLayout(inner)
        self.thread.setContentsMargins(0, 0, 0, 0)
        self.thread.setSpacing(28)
        self.thread.addStretch(1)
        wrap.addStretch(1)
        wrap.addWidget(inner, 4)
        wrap.addStretch(1)
        self.thread_scroll.setWidget(holder)
        col.addWidget(self.thread_scroll, 1)

        self._opening = self._build_opening()
        self.thread.insertWidget(0, self._opening)

        col.addWidget(self._build_bar())
        return main

    def _build_opening(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 40, 0, 0)
        v.setSpacing(14)
        h = QLabel("Ask this patient's sessions anything.")
        h.setObjectName("Headline")
        h.setWordWrap(True)
        lede = QLabel(
            "Answers come only from what was actually said — and every one of them "
            "points back to the second it was said on."
        )
        lede.setObjectName("Lede")
        lede.setWordWrap(True)
        v.addWidget(h)
        v.addWidget(lede)

        seeds = QHBoxLayout()
        seeds.setSpacing(7)
        for text in SEEDS:
            b = QPushButton(text)
            b.setObjectName("Seed")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, t=text: self._send(t))
            seeds.addWidget(b)
        seeds.addStretch(1)
        v.addLayout(seeds)
        return w

    def _build_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("Bar")
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(40, 14, 40, 16)
        outer.setSpacing(7)

        centred = QHBoxLayout()
        frame = QFrame()
        frame.setObjectName("AskFrame")
        frame.setMaximumWidth(700)
        f = QHBoxLayout(frame)
        f.setContentsMargins(13, 7, 7, 7)
        f.setSpacing(8)

        self.ask = QPlainTextEdit()
        self.ask.setObjectName("Ask")
        self.ask.setPlaceholderText("Ask about this patient…")
        self.ask.setFixedHeight(38)
        self.ask.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ask.textChanged.connect(self._grow_ask)
        self.ask.installEventFilter(self)

        self.send = QPushButton("↑")
        self.send.setObjectName("Send")
        self.send.setCursor(Qt.PointingHandCursor)
        self.send.clicked.connect(lambda: self._send(self.ask.toPlainText()))

        f.addWidget(self.ask, 1)
        f.addWidget(self.send, 0, Qt.AlignBottom)
        centred.addStretch(1)
        centred.addWidget(frame, 4)
        centred.addStretch(1)
        outer.addLayout(centred)

        hint_row = QHBoxLayout()
        hint = QLabel("Enter to send · Shift+Enter for a new line")
        hint.setObjectName("Searched")
        self.ctx = QLabel("no sessions indexed")
        self.ctx.setObjectName("Searched")
        hint_row.addStretch(1)
        holder = QWidget()
        holder.setMaximumWidth(700)
        hh = QHBoxLayout(holder)
        hh.setContentsMargins(0, 0, 0, 0)
        hh.addWidget(hint)
        hh.addStretch(1)
        hh.addWidget(self.ctx)
        hint_row.addWidget(holder, 4)
        hint_row.addStretch(1)
        outer.addLayout(hint_row)
        return bar

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent

        if obj is self.ask and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (
                event.modifiers() & Qt.ShiftModifier
            ):
                self._send(self.ask.toPlainText())
                return True
        return super().eventFilter(obj, event)

    def _grow_ask(self) -> None:
        doc = self.ask.document().size().height()
        self.ask.setFixedHeight(int(min(140, max(38, doc + 12))))

    # ======================================================================
    # recording
    # ======================================================================
    def _toggle_record(self) -> None:
        if self.recorder.recording:
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self) -> None:
        if self._ingest and self._ingest.isRunning():
            return self._warn("Still processing the last session. One at a time.")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = config.AUDIO_DIR / f"p{self.pid.value()}_{stamp}.wav"
        try:
            self.recorder.start(path)
        except Exception as exc:
            return self._warn(f"Couldn't open the microphone: {exc}")

        self._warn(None)
        self.rec.set_recording(True)
        self.meter.set_live(True)
        self.pick.setEnabled(False)
        self.pid.setEnabled(False)
        self.alias.setEnabled(False)
        self._timer.start(50)

    def _stop_record(self) -> None:
        self._timer.stop()
        self.rec.set_recording(False)
        self.meter.set_live(False)
        try:
            path = self.recorder.stop()
        except Exception as exc:
            self._reset_capture()
            return self._warn(f"The recording didn't close cleanly: {exc}")

        if self.recorder.seconds < 1.0:
            path.unlink(missing_ok=True)
            self._reset_capture()
            return self._warn("That recording was under a second. Nothing was saved.")

        self._process(path)

    def _on_level(self, level: float) -> None:
        self.meter.push(level)

    def _tick(self) -> None:
        self.clock.setText(mmss(self.recorder.seconds))
        self._pulse = (self._pulse + 0.03) % 1.0
        self.rec.set_pulse(self._pulse)

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a recorded session", str(Path.home()),
            "Audio (*.wav *.flac *.ogg *.mp3 *.m4a *.aiff);;All files (*)",
        )
        if path:
            self._process(Path(path))

    def _reset_capture(self) -> None:
        self.clock.setText("00:00")
        self.rec.setEnabled(True)
        self.pick.setEnabled(True)
        self.pid.setEnabled(True)
        self.alias.setEnabled(True)

    # ======================================================================
    # processing
    # ======================================================================
    def _process(self, audio: Path) -> None:
        # A file dragged in from elsewhere has to be copied into the vault's
        # audio directory: WhisperX derives its transcript path from the audio
        # file's grandparent, and the citation player needs the file to still be
        # there next week.
        if audio.parent != config.AUDIO_DIR:
            dest = config.AUDIO_DIR / f"p{self.pid.value()}_{audio.name}"
            if not dest.exists():
                dest.write_bytes(audio.read_bytes())
            audio = dest

        self.work.show()
        self.log.clear()
        self.rail.set_stage("preparing")
        self.rec.setEnabled(False)
        self.pick.setEnabled(False)
        self.pid.setEnabled(False)
        self.alias.setEnabled(False)

        self._ingest = IngestWorker(
            audio, self._passphrase, self.pid.value(), self.alias.text(),
            {"SPEAKER_00": "Patient", "SPEAKER_01": "Therapist"},
        )
        self._ingest.stage.connect(self.rail.set_stage)
        self._ingest.log.connect(self._append_log)
        self._ingest.finished_ok.connect(self._ingest_done)
        self._ingest.failed.connect(self._ingest_failed)
        self._ingest.start()

    def _append_log(self, line: str) -> None:
        if line.strip():
            self.log.appendPlainText(line.strip())

    def _ingest_done(self, result) -> None:
        self.rail.set_stage("done")
        self._reset_capture()
        self._reload_sessions()
        QTimer.singleShot(2500, self.work.hide)

    def _ingest_failed(self, message: str) -> None:
        self._reset_capture()
        self.work.hide()
        self._warn(message)

    # ======================================================================
    # sessions
    # ======================================================================
    def _on_patient_changed(self) -> None:
        self._history.clear()
        self._clear_thread()
        self._reload_sessions()

    def _reload_sessions(self) -> None:
        self._sessions = vault.sessions(self.pid.value())

        while self.slist.count() > 1:
            item = self.slist.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._sessions:
            empty = QLabel(
                "Nothing recorded for this patient yet. Press record, or open an "
                "audio file you already have."
            )
            empty.setObjectName("Empty")
            empty.setWordWrap(True)
            self.slist.insertWidget(0, empty)
            self.ctx.setText("no sessions indexed")
            return

        n = len(self._sessions)
        self.ctx.setText(f"{n} session{'s' if n > 1 else ''} · searching all")
        for i, s in enumerate(self._sessions):
            card = QPushButton()
            card.setObjectName("SessionCard")
            card.setCursor(Qt.PointingHandCursor)
            card.setFlat(True)
            card.setMinimumHeight(74)
            v = QVBoxLayout(card)
            v.setContentsMargins(12, 10, 12, 10)
            v.setSpacing(5)
            top = QHBoxLayout()
            title = QLabel(f"Session {s['id']}")
            title.setStyleSheet("font-weight:600;font-size:13px;")
            meta = QLabel(f"{mmss(s['duration'])} · {s['chunks']} chunks")
            meta.setObjectName("Searched")
            top.addWidget(title)
            top.addStretch(1)
            top.addWidget(meta)
            note = QLabel((s["note"] or "No note").strip()[:150].replace("\n", " "))
            note.setObjectName("CiteText")
            note.setWordWrap(True)
            note.setMaximumHeight(34)
            v.addLayout(top)
            v.addWidget(note)
            card.clicked.connect(lambda _=False, sid=s["id"]: self._play_session(sid))
            self.slist.insertWidget(i, card)

    def _play_session(self, session_id: int) -> None:
        s = vault.session(session_id)
        if s:
            self.player.play_at(Path(s["audio_path"]), 0.0)

    def _audio_for(self, session_id: int) -> Path | None:
        s = vault.session(session_id)
        return Path(s["audio_path"]) if s else None

    def _duration_of(self, session_id: int) -> float:
        for s in self._sessions:
            if s["id"] == session_id:
                return s["duration"]
        s = vault.session(session_id)
        return s["duration"] if s else 1.0

    # ======================================================================
    # chat
    # ======================================================================
    def _clear_thread(self) -> None:
        while self.thread.count() > 1:
            item = self.thread.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._opening = self._build_opening()
        self.thread.insertWidget(0, self._opening)

    def _send(self, text: str) -> None:
        question = (text or "").strip()
        if not question:
            return
        if self._chat and self._chat.isRunning():
            return
        if not self._sessions:
            return self._warn("There's nothing indexed for this patient to search yet.")

        self._warn(None)
        self.ask.clear()
        self.send.setEnabled(False)

        if self._opening is not None:
            self._opening.deleteLater()
            self._opening = None

        turn = self._new_turn(question)
        self.thread.insertWidget(self.thread.count() - 1, turn["widget"])
        self._scroll_down()

        self._chat = ChatWorker(self.pid.value(), question, self._history)
        self._chat.sources.connect(lambda p, t=turn: self._render_evidence(t, p))
        self._chat.token.connect(lambda tok, t=turn: self._render_token(t, tok))
        self._chat.finished_ok.connect(lambda ans, q=question: self._chat_done(q, ans))
        self._chat.failed.connect(lambda m, t=turn: self._chat_failed(t, m))
        self._chat.start()

    def _new_turn(self, question: str) -> dict:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        v.addWidget(_eyebrow("You asked"))
        q = QLabel(question)
        q.setObjectName("Question")
        q.setWordWrap(True)
        v.addWidget(q)

        spacer = _eyebrow("From the record")
        spacer.setContentsMargins(0, 12, 0, 0)
        v.addWidget(spacer)

        a = QLabel("…")
        a.setObjectName("Answer")
        a.setWordWrap(True)
        a.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(a)

        evidence = QWidget()
        ev = QVBoxLayout(evidence)
        ev.setContentsMargins(0, 12, 0, 0)
        ev.setSpacing(9)
        v.addWidget(evidence)

        return {"widget": w, "answer": a, "evidence": ev, "text": ""}

    def _render_token(self, turn: dict, token: str) -> None:
        turn["text"] += token
        turn["answer"].setText(turn["text"])
        self._scroll_down()

    def _render_evidence(self, turn: dict, payload: dict) -> None:
        sources = payload["sources"]
        box = turn["evidence"]

        line = QFrame()
        line.setObjectName("Evidence")
        line.setFrameShape(QFrame.HLine)
        line.setFixedHeight(1)
        box.addWidget(line)

        if not sources:
            none = QLabel("Nothing in this patient's sessions matched.")
            none.setObjectName("Empty")
            none.setWordWrap(True)
            box.addWidget(none)
            return

        box.addWidget(_eyebrow(f"Evidence · {len(sources)} excerpt"
                              f"{'s' if len(sources) > 1 else ''}"))
        if payload["search_query"]:
            sq = QLabel(f"searched: {payload['search_query']}")
            sq.setObjectName("Searched")
            sq.setWordWrap(True)
            box.addWidget(sq)

        by_session: dict[int, list[dict]] = {}
        for s in sources:
            by_session.setdefault(s["session_id"], []).append(s)

        for sid, hits in by_session.items():
            total = self._duration_of(sid)
            audio = self._audio_for(sid)

            head = QHBoxLayout()
            left = QLabel(f"Session {sid}")
            left.setObjectName("RibbonLabel")
            right = QLabel(mmss(total))
            right.setObjectName("RibbonLabel")
            head.addWidget(left)
            head.addStretch(1)
            head.addWidget(right)
            box.addLayout(head)

            ribbon = Ribbon(total, hits)
            box.addWidget(ribbon)

            cites: list[QPushButton] = []
            for i, h in enumerate(hits):
                cite = QPushButton()
                cite.setFlat(True)
                cite.setCursor(Qt.PointingHandCursor)
                cite.setStyleSheet("text-align:left;border:none;padding:4px 6px;")
                cl = QHBoxLayout(cite)
                cl.setContentsMargins(6, 4, 6, 4)
                cl.setSpacing(10)
                ts = QLabel(h["start_ts"])
                ts.setObjectName("CiteTime")
                ts.setFixedWidth(38)
                txt = QLabel(h["text"].replace("\n", "  ").strip())
                txt.setObjectName("CiteText")
                txt.setWordWrap(True)
                cl.addWidget(ts, 0, Qt.AlignTop)
                cl.addWidget(txt, 1)
                cite.setMinimumHeight(38)
                box.addWidget(cite)
                cites.append(cite)

                def jump(_=False, idx=i, rb=ribbon, hh=hits, ap=audio):
                    rb.set_selected(idx)
                    if ap:
                        self.player.play_at(ap, hh[idx]["start"])

                cite.clicked.connect(jump)

            if audio:
                ribbon.seek.connect(lambda sec, ap=audio: self.player.play_at(ap, sec))
                self.player.player.positionChanged.connect(
                    lambda ms, rb=ribbon, ap=audio: (
                        rb.set_playhead(ms / 1000.0)
                        if self.player.source == ap else None
                    )
                )

        self._scroll_down()

    def _chat_done(self, question: str, answer: str) -> None:
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer})
        self.send.setEnabled(True)
        self.ask.setFocus()

    def _chat_failed(self, turn: dict, message: str) -> None:
        turn["answer"].setText(f"That didn't work: {message}")
        self.send.setEnabled(True)

    def _scroll_down(self) -> None:
        bar = self.thread_scroll.verticalScrollBar()
        QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))

    # ======================================================================
    def _warn(self, message: str | None) -> None:
        if not message:
            self.banner.hide()
            return
        self.banner.setText(message)
        self.banner.show()

    def closeEvent(self, event) -> None:
        if self._ingest and self._ingest.isRunning():
            answer = QMessageBox.question(
                self, "Still processing",
                "A session is still being transcribed. Quit anyway and lose it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self._ingest.cancel()
        if self.recorder.recording:
            self.recorder.stop()
        self.player.stop()
        vault.lock()
        event.accept()
