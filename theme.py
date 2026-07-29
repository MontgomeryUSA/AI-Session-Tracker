"""
theme.py
========
The look, in one file.

Two constraints shaped this:

  * No webfonts. A <link> to fonts.googleapis.com is a network call, and this
    app doesn't make those. Fonts either ship in the bundle or we use what the
    OS already has. resolve_fonts() tries the bundle first and degrades to a
    native stack, so a missing TTF costs polish, not a crash.

  * Clinical, not consumer. Cool grey paper, ink, and exactly two accents that
    each mean something: red is live audio, teal is retrieved evidence. Nothing
    else gets colour, so when something IS coloured, it's telling you something.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QFontDatabase

import config

# ---------------------------------------------------------------------------
PAPER    = "#E9EDF0"   # the desk
CARD     = "#FCFDFE"   # things on the desk
INK      = "#14212B"
INK_2    = "#61727E"   # secondary text
RULE     = "#CFD8DE"   # hairlines
LIVE     = "#B8342A"   # recording, and errors. the only red in the building
EVIDENCE = "#0E7C6B"   # a chunk the model actually used
FOCUS    = "#0E7C6B"

Q_LIVE = QColor(LIVE)
Q_EVIDENCE = QColor(EVIDENCE)
Q_RULE = QColor(RULE)
Q_PAPER = QColor(PAPER)
Q_INK = QColor(INK)
Q_INK2 = QColor(INK_2)
Q_CARD = QColor(CARD)

FONTS_DIR = config.BUNDLE / "assets" / "fonts"

UI_FAMILY = "Inter"
DOC_FAMILY = "Source Serif 4"
DATA_FAMILY = "JetBrains Mono"


def resolve_fonts() -> None:
    """Load bundled TTFs if they're there; otherwise fall back to the platform."""
    global UI_FAMILY, DOC_FAMILY, DATA_FAMILY

    loaded: set[str] = set()
    if FONTS_DIR.is_dir():
        for ttf in FONTS_DIR.glob("*.tt[fc]"):
            fid = QFontDatabase.addApplicationFont(str(ttf))
            if fid != -1:
                loaded.update(QFontDatabase.applicationFontFamilies(fid))

    available = set(QFontDatabase.families()) | loaded

    def pick(*candidates: str) -> str:
        for c in candidates:
            if c in available:
                return c
        return candidates[-1]

    UI_FAMILY = pick("Inter", "Archivo", "Segoe UI", "Helvetica Neue", "sans-serif")
    DOC_FAMILY = pick("Source Serif 4", "Charter", "Georgia", "serif")
    DATA_FAMILY = pick("JetBrains Mono", "SF Mono", "Cascadia Mono",
                       "DejaVu Sans Mono", "monospace")


def qss() -> str:
    return f"""
    QWidget {{
        background: {PAPER};
        color: {INK};
        font-family: "{UI_FAMILY}";
        font-size: 13px;
    }}

    /* ---------- structure ---------- */
    #Rail {{
        background: {CARD};
        border-right: 1px solid {RULE};
    }}
    #Brand    {{ border-bottom: 1px solid {RULE}; }}
    #Capture  {{ border-bottom: 1px solid {RULE}; }}
    #Work     {{ border-bottom: 1px solid {RULE}; }}
    #Bar      {{ background: {CARD}; border-top: 1px solid {RULE}; }}

    #BrandName {{ font-size: 15px; font-weight: 700; }}
    #Eyebrow, #BrandSub {{
        font-family: "{DATA_FAMILY}";
        font-size: 10px;
        color: {INK_2};
        letter-spacing: 1px;
    }}

    /* ---------- inputs ---------- */
    QLineEdit, QSpinBox {{
        font-family: "{DATA_FAMILY}";
        font-size: 13px;
        padding: 6px 8px;
        border: 1px solid {RULE};
        border-radius: 5px;
        background: {PAPER};
        selection-background-color: {EVIDENCE};
        selection-color: {CARD};
    }}
    QLineEdit:focus, QSpinBox:focus {{ border-color: {FOCUS}; }}
    QLineEdit:disabled, QSpinBox:disabled {{ color: {INK_2}; }}

    #Ask {{
        border: none;
        background: transparent;
        font-family: "{DOC_FAMILY}";
        font-size: 16px;
        padding: 4px 0;
    }}
    #AskFrame {{
        background: {PAPER};
        border: 1px solid {RULE};
        border-radius: 10px;
    }}
    #AskFrame[focused="true"] {{ border-color: {FOCUS}; }}

    /* ---------- buttons ---------- */
    #Send {{
        background: {INK};
        color: {CARD};
        border-radius: 7px;
        min-width: 34px; max-width: 34px;
        min-height: 34px; max-height: 34px;
        font-size: 15px;
    }}
    #Send:hover    {{ background: {EVIDENCE}; }}
    #Send:disabled {{ background: {INK_2}; }}

    #Link {{
        border: none;
        background: transparent;
        color: {INK_2};
        font-family: "{DATA_FAMILY}";
        font-size: 11px;
        text-decoration: underline;
        padding: 2px;
    }}
    #Link:hover {{ color: {EVIDENCE}; }}

    #Seed {{
        border: 1px solid {RULE};
        border-radius: 14px;
        padding: 6px 13px;
        background: {CARD};
        color: {INK_2};
        font-family: "{DOC_FAMILY}";
        font-size: 13px;
    }}
    #Seed:hover {{ border-color: {EVIDENCE}; color: {EVIDENCE}; }}

    #SessionCard {{
        border: 1px solid {RULE};
        border-radius: 7px;
        background: {CARD};
        text-align: left;
        padding: 11px 12px;
    }}
    #SessionCard:hover {{ border-color: {EVIDENCE}; }}
    #SessionCard[current="true"] {{ border-color: {EVIDENCE}; background: {PAPER}; }}

    /* ---------- chat ---------- */
    #Headline {{ font-size: 26px; font-weight: 700; }}
    #Lede, #Empty {{
        font-family: "{DOC_FAMILY}";
        font-size: 15px;
        color: {INK_2};
    }}
    #Question {{
        font-family: "{DOC_FAMILY}";
        font-size: 17px;
        font-weight: 600;
    }}
    #Answer {{
        font-family: "{DOC_FAMILY}";
        font-size: 16px;
        line-height: 160%;
    }}
    #Searched, #CiteTime, #RibbonLabel {{
        font-family: "{DATA_FAMILY}";
        font-size: 10px;
        color: {INK_2};
    }}
    #CiteTime {{ color: {EVIDENCE}; }}
    #CiteText {{
        font-family: "{DOC_FAMILY}";
        font-size: 13px;
        color: {INK_2};
    }}
    #Evidence {{ border-top: 1px solid {RULE}; }}

    /* ---------- progress ---------- */
    #Log {{
        background: {PAPER};
        border: 1px solid {RULE};
        border-radius: 5px;
        font-family: "{DATA_FAMILY}";
        font-size: 10px;
        color: {INK_2};
        padding: 6px;
    }}
    #Step        {{ color: {INK_2}; font-size: 12px; }}
    #Step[state="active"] {{ color: {INK}; font-weight: 600; }}
    #Step[state="done"]   {{ color: {INK}; }}
    #StepNum     {{ font-family: "{DATA_FAMILY}"; font-size: 10px; color: {RULE}; }}
    #StepNum[state="active"] {{ color: {LIVE}; }}
    #StepNum[state="done"]   {{ color: {EVIDENCE}; }}

    #Banner {{ background: {LIVE}; color: white; padding: 8px 12px; font-size: 12px; }}

    /* ---------- scrollbars ---------- */
    QScrollArea, QScrollArea > QWidget > QWidget {{ border: none; background: transparent; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
    QScrollBar::handle:vertical {{
        background: {RULE}; border-radius: 5px; min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {INK_2}; }}
    QScrollBar::add-line, QScrollBar::sub-line,
    QScrollBar::add-page, QScrollBar::sub-page {{ height: 0; background: none; border: none; }}

    QToolTip {{
        background: {INK}; color: {CARD}; border: none;
        padding: 5px 7px; font-family: "{DATA_FAMILY}"; font-size: 10px;
    }}
    """
