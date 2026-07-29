"""
run_app.py
==========
Entry point. The order matters more than it looks.

  1. import config          sets HF offline env BEFORE anything imports whisperx
  2. install_network_guard  from this line on, opening ANY IP socket raises
  3. warm the models        loaded from disk, into this process. No server.
  4. unlock the vault       passphrase in memory, never on disk
  5. open the window

After step 2 this process is incapable of touching a network. Not "doesn't", not
"shouldn't" -- can't. There is no daemon, no port, no loopback client. Everything
from here is files on this disk and models in this process's memory.

Launch:  python run_app.py
"""
from __future__ import annotations

import sys
import traceback

import config                       # 1. must be first: sets offline env at import
config.install_network_guard()      # 2. door closed, and welded

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QVBoxLayout,
)

import local_llm
import theme
import vault


class WarmUpDialog(QDialog):
    """Loading a 4B chat model plus bge-m3 off disk takes a few seconds and pins
    a core. Doing it lazily would mean the clinician's FIRST question -- or worse,
    the SOAP note at the end of a real session -- silently stalls. So pay it here,
    once, where a progress bar makes it legible."""

    class _Loader(QThread):
        step = Signal(str)
        done = Signal()
        failed = Signal(str)

        def run(self):
            try:
                local_llm.warm_up(self.step.emit)
                self.done.emit()
            except Exception as exc:
                self.failed.emit(str(exc))

    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        self.setFixedWidth(400)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)

        v = QVBoxLayout(self)
        v.setContentsMargins(28, 26, 28, 24)
        v.setSpacing(12)

        title = QLabel(config.APP_NAME)
        title.setObjectName("BrandName")
        v.addWidget(title)

        self.status = QLabel("Starting…")
        self.status.setObjectName("Lede")
        self.status.setWordWrap(True)
        v.addWidget(self.status)

        bar = QProgressBar()
        bar.setRange(0, 0)          # indeterminate: llama.cpp won't tell us a %
        bar.setTextVisible(False)
        bar.setFixedHeight(4)
        v.addWidget(bar)

        self.error: str | None = None
        self._loader = self._Loader()
        self._loader.step.connect(self.status.setText)
        self._loader.done.connect(self.accept)
        self._loader.failed.connect(self._fail)
        self._loader.start()

    def _fail(self, message: str) -> None:
        self.error = message
        self.reject()


class UnlockDialog(QDialog):
    """The vault is SQLCipher-encrypted. The passphrase lives in memory for the
    life of the process and is written nowhere -- not a config file, not an
    environment variable, not the keychain. Close the app and it's gone.

    This is what replaces `PASSPHRASE = "password"` on line 30 of chat_session.py.
    """

    def __init__(self, first_run: bool):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        self.setFixedWidth(400)
        self.setModal(True)

        v = QVBoxLayout(self)
        v.setContentsMargins(28, 26, 28, 22)
        v.setSpacing(12)

        title = QLabel("Create a passphrase" if first_run else "Unlock the vault")
        title.setObjectName("BrandName")
        v.addWidget(title)

        body = QLabel(
            "This encrypts every transcript, note, and recording index on this "
            "computer. There is no recovery — if you lose it, the sessions are gone."
            if first_run else
            "Enter the passphrase for this computer's session vault."
        )
        body.setObjectName("Lede")
        body.setWordWrap(True)
        v.addWidget(body)

        self.field = QLineEdit()
        self.field.setEchoMode(QLineEdit.Password)
        self.field.setPlaceholderText("Passphrase")
        self.field.returnPressed.connect(self.accept)
        v.addWidget(self.field)

        self.error = QLabel()
        self.error.setStyleSheet(f"color:{theme.LIVE};font-size:12px;")
        self.error.setWordWrap(True)
        self.error.hide()
        v.addWidget(self.error)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    def accept(self) -> None:
        passphrase = self.field.text()
        if not passphrase:
            return self._fail("Enter a passphrase.")
        try:
            vault.unlock(passphrase)
        except vault.Locked as exc:
            return self._fail(str(exc))
        except Exception as exc:
            return self._fail(f"Couldn't open the vault: {exc}")
        self.passphrase = passphrase
        super().accept()

    def _fail(self, message: str) -> None:
        self.error.setText(message)
        self.error.show()
        self.field.selectAll()
        self.field.setFocus()


def fatal(title: str, detail: str) -> None:
    box = QMessageBox()
    box.setIcon(QMessageBox.Critical)
    box.setWindowTitle(config.APP_NAME)
    box.setText(title)
    box.setInformativeText(detail)
    box.exec()
    sys.exit(1)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(config.APP_NAME)

    theme.resolve_fonts()
    app.setStyleSheet(theme.qss())

    # 3. models, into this process
    warm = WarmUpDialog()
    if warm.exec() != QDialog.Accepted:
        fatal("The models couldn't be loaded.", warm.error or "Cancelled.")

    # 4. vault
    first_run = not config.DB_PATH.exists()
    unlock = UnlockDialog(first_run)
    if unlock.exec() != QDialog.Accepted:
        sys.exit(0)

    # 5. window  (imported late: it expects an unlocked vault)
    from main_window import MainWindow

    window = MainWindow(unlock.passphrase)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
