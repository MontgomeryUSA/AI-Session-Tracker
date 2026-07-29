"""
widgets.py
==========
The hand-painted parts.

The signature here is the Ribbon. Everything this app knows is anchored to a
moment in a recording -- recall() hands back (start, end) for every chunk it
retrieves -- so an answer isn't just text, it has a shape in time. The Ribbon
draws each session as a bar, lights the excerpts the model actually used, and
clicking one seeks the audio to that second. You can hear the sentence the model
is citing. That is the whole argument for building this on top of a transcript
rather than a summary, made visible.

Everything else on screen is deliberately quiet so the ribbon is the thing you
remember.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import QLabel, QPushButton, QSizePolicy, QWidget

import theme


def mmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


# ---------------------------------------------------------------------------
class RecordButton(QPushButton):
    """A disc that becomes a rounded square while recording, with a pulse ring.

    Universal recorder grammar. Nobody needs a legend for it.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(62, 62)
        self.setCursor(Qt.PointingHandCursor)
        self.setFlat(True)
        self._on = False
        self._pulse = 0.0
        self.setAccessibleName("Start recording")

    def set_recording(self, on: bool) -> None:
        self._on = on
        self.setAccessibleName("Stop recording" if on else "Start recording")
        self.update()

    def set_pulse(self, phase: float) -> None:
        self._pulse = phase
        if self._on:
            self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(11, 11, 40, 40)

        if self._on:
            ring = QRectF(r).adjusted(-8 * self._pulse, -8 * self._pulse,
                                      8 * self._pulse, 8 * self._pulse)
            pen = QPen(QColor(theme.LIVE))
            pen.setWidthF(1.5)
            c = QColor(theme.LIVE)
            c.setAlphaF(max(0.0, 0.55 * (1.0 - self._pulse)))
            pen.setColor(c)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(ring)

        p.setPen(Qt.NoPen)
        if not self.isEnabled():
            p.setBrush(QBrush(QColor(theme.INK_2).lighter(160)))
        else:
            p.setBrush(QBrush(QColor(theme.LIVE)))

        if self._on:
            p.drawRoundedRect(r.adjusted(3, 3, -3, -3), 7, 7)  # square = stop
        else:
            p.drawEllipse(r)


# ---------------------------------------------------------------------------
class LevelMeter(QWidget):
    """Live input level. Not decoration -- it's the only proof the microphone is
    actually picking the room up, which is the one thing you cannot recover from
    getting wrong."""

    BARS = 26

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(62)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._levels = [0.0] * self.BARS
        self._live = False

    def set_live(self, live: bool) -> None:
        self._live = live
        if not live:
            self._levels = [0.0] * self.BARS
        self.update()

    def push(self, level: float) -> None:
        self._levels = self._levels[1:] + [max(0.0, min(1.0, level))]
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width() / self.BARS
        mid = self.height() / 2
        colour = QColor(theme.LIVE) if self._live else QColor(theme.RULE)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(colour))
        for i, lv in enumerate(self._levels):
            h = 3 + lv * 40
            p.drawRoundedRect(QRectF(i * w + 1, mid - h / 2, max(1.5, w - 2), h), 1, 1)


# ---------------------------------------------------------------------------
class StepRail(QWidget):
    """The four stages, numbered -- because this genuinely is a sequence and the
    order tells you how far off the finish line you are. (A numbered list that
    isn't a sequence is just decoration.)"""

    STEPS = [
        ("preparing",    "Resample to 16 kHz mono"),
        ("transcribing", "Transcribe & separate speakers"),
        ("summarizing",  "Draft the SOAP note"),
        ("indexing",     "Embed & encrypt into the vault"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout

        self._rows = []
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        for i, (key, label) in enumerate(self.STEPS, start=1):
            row = QHBoxLayout()
            row.setSpacing(10)
            num = QLabel(str(i))
            num.setObjectName("StepNum")
            num.setFixedWidth(14)
            text = QLabel(label)
            text.setObjectName("Step")
            row.addWidget(num)
            row.addWidget(text, 1)
            col.addLayout(row)
            self._rows.append((key, num, text))

    def set_stage(self, stage: str) -> None:
        keys = [k for k, _ in self.STEPS]
        idx = keys.index(stage) if stage in keys else (len(keys) if stage == "done" else -1)
        for i, (_key, num, text) in enumerate(self._rows):
            state = "done" if i < idx else ("active" if i == idx else "idle")
            for w in (num, text):
                w.setProperty("state", state)
                w.style().unpolish(w)
                w.style().polish(w)
            num.setText("✓" if state == "done" else str(i + 1))


# ---------------------------------------------------------------------------
class Ribbon(QWidget):
    """One session, drawn as its own duration. Retrieved chunks light up on it.

    Click a lit span to hear it.
    """

    seek = Signal(float)      # seconds
    picked = Signal(int)      # index into self._spans

    def __init__(self, duration: float, spans: list[dict], parent=None):
        super().__init__(parent)
        self.setFixedHeight(24)
        self.setCursor(Qt.PointingHandCursor)
        self._duration = max(1.0, float(duration))
        self._spans = spans
        self._selected = -1
        self._playhead = -1.0
        self.setMouseTracking(True)

    def set_selected(self, i: int) -> None:
        self._selected = i
        self.update()

    def set_playhead(self, seconds: float) -> None:
        self._playhead = seconds
        self.update()

    def _x(self, seconds: float) -> float:
        return (seconds / self._duration) * self.width()

    def _hit(self, x: float) -> int:
        for i, s in enumerate(self._spans):
            if self._x(s["start"]) - 3 <= x <= self._x(s["end"]) + 3:
                return i
        return -1

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)

        path = QPainterPath()
        path.addRoundedRect(rect, 3, 3)
        p.setClipPath(path)

        p.fillPath(path, QBrush(QColor(theme.PAPER)))

        for i, s in enumerate(self._spans):
            x0 = self._x(s["start"])
            x1 = max(x0 + 2.0, self._x(s["end"]))
            fill = QColor(theme.EVIDENCE)
            fill.setAlphaF(0.62 if i == self._selected else 0.26)
            p.fillRect(QRectF(x0, 0, x1 - x0, self.height()), fill)
            p.setPen(QPen(QColor(theme.EVIDENCE), 2))
            p.drawLine(QPointF(x0 + 1, 0), QPointF(x0 + 1, self.height()))

        if self._playhead >= 0:
            p.setPen(QPen(QColor(theme.LIVE), 1))
            x = self._x(self._playhead)
            p.drawLine(QPointF(x, 0), QPointF(x, self.height()))

        p.setClipping(False)
        p.setPen(QPen(QColor(theme.RULE), 1))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

    def mousePressEvent(self, e) -> None:
        i = self._hit(e.position().x())
        if i >= 0:
            self.set_selected(i)
            self.picked.emit(i)
            self.seek.emit(self._spans[i]["start"])

    def mouseMoveEvent(self, e) -> None:
        i = self._hit(e.position().x())
        if i >= 0:
            s = self._spans[i]
            self.setToolTip(f'{mmss(s["start"])}–{mmss(s["end"])}')
        else:
            self.setToolTip("")


# ---------------------------------------------------------------------------
class AudioPlayer:
    """One player for the whole window. Two answers citing the same session share
    it, so you never get two recordings talking over each other."""

    def __init__(self, parent: QWidget):
        self._out = QAudioOutput(parent)
        self.player = QMediaPlayer(parent)
        self.player.setAudioOutput(self._out)
        self._source: Path | None = None

    def play_at(self, path: Path, seconds: float) -> None:
        path = Path(path)
        if not path.exists():
            return
        if self._source != path:
            self.player.setSource(QUrl.fromLocalFile(str(path)))
            self._source = path
        self.player.setPosition(int(seconds * 1000))
        self.player.play()

    def stop(self) -> None:
        self.player.stop()

    @property
    def source(self) -> Path | None:
        return self._source
