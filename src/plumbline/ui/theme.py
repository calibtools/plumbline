"""Calib.Tools visual identity for the desktop app: colours, type, style sheet."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase

ASSETS = Path(__file__).resolve().parent.parent / "assets"

PAPER = "#F4F3EF"
PAPER_ALT = "#EAE8E2"
SHEET = "#FFFFFF"
INK = "#141414"
INK_MUTED = "#4A4A47"
INK_SOFT = "#6B6A66"
HAIRLINE = "#D6D3CB"
LINE_STRONG = "#A8A59C"
RED = "#E60012"  # fills and large marks only
RED_TEXT = "#B3000E"
RED_TINT = "#FBE9EA"
GREEN_TEXT = "#007A3D"

SITE = "https://calib.tools"
VISIT = "Visit Calib.Tools ↗"



def site_url(path: str = "", where: str = "app") -> str:
    """Address on calib.tools, marked so the site can tell the visit came from Plumbline.

    The mark says only which place in the app the link sits in. Nothing about the user is added.
    """
    return f"{SITE}{path}?utm_source=plumbline&utm_medium=app&utm_content={where}"


SANS = "Geist"
MONO = "Geist Mono"


def asset(name: str) -> str:
    return str(ASSETS / name)


def load_fonts() -> None:
    for f in sorted((ASSETS / "fonts").glob("*.otf")):
        QFontDatabase.addApplicationFont(str(f))


def sans(px: int, weight: int = 400) -> QFont:
    f = QFont(SANS)
    f.setPixelSize(px)
    f.setWeight(QFont.Weight(weight))
    return f


def mono(px: int, weight: int = 400) -> QFont:
    f = QFont(MONO)
    f.setPixelSize(px)
    f.setWeight(QFont.Weight(weight))
    return f


STYLE = f"""
* {{ font-family: "{SANS}"; font-size: 14px; color: {INK}; }}
QMainWindow, QWidget#page {{ background: {PAPER}; }}
QLabel {{ background: transparent; }}
QLabel[role="eyebrow"] {{ font-family: "{MONO}"; font-size: 12px; font-weight: 500; color: {INK_MUTED}; }}
QLabel[role="muted"] {{ color: {INK_MUTED}; font-size: 13px; }}
QLabel[role="small"] {{ color: {INK_MUTED}; font-size: 12px; }}
QLabel[role="link"] {{ color: {RED_TEXT}; font-size: 13px; }}
QLabel[role="monolink"] {{ color: {RED_TEXT}; font-family: "{MONO}"; font-size: 13px; font-weight: 600; }}

QPushButton {{
    min-height: 38px; padding: 0 16px; border: 1px solid {LINE_STRONG}; border-radius: 0;
    background: transparent; font-family: "{MONO}"; font-size: 13px; font-weight: 600; text-align: left;
}}
QPushButton:hover {{ border-color: {INK}; }}
QPushButton:disabled {{ color: {INK_SOFT}; border-color: {HAIRLINE}; }}
QPushButton[kind="primary"] {{ background: {RED}; border-color: {RED}; color: #FFFFFF; }}
QPushButton[kind="primary"]:hover {{ background: {RED_TEXT}; border-color: {RED_TEXT}; }}
QPushButton[kind="primary"]:disabled {{ background: {PAPER_ALT}; border-color: {HAIRLINE}; color: {INK_SOFT}; }}
QPushButton[kind="ink"] {{ border-color: {INK}; }}
QPushButton[kind="big"] {{ min-height: 46px; font-size: 14px; }}
QPushButton[kind="bigprimary"] {{ min-height: 46px; font-size: 14px; background: {RED}; border-color: {RED}; color: #FFFFFF; }}
QPushButton[kind="bigprimary"]:hover {{ background: {RED_TEXT}; border-color: {RED_TEXT}; }}

QFrame[role="sheet"] {{ background: {SHEET}; border: 1px solid {HAIRLINE}; }}
QFrame[role="inksheet"] {{ background: {SHEET}; border: 1px solid {INK}; }}
QFrame[role="rule"] {{ background: {HAIRLINE}; max-height: 1px; min-height: 1px; border: 0; }}
QFrame[role="vrule"] {{ background: {HAIRLINE}; max-width: 1px; min-width: 1px; border: 0; }}
QFrame[role="inkrule"] {{ background: {INK}; max-height: 1px; min-height: 1px; border: 0; }}

QSpinBox, QDoubleSpinBox {{
    padding-left: 10px; border: 1px solid {LINE_STRONG}; border-radius: 0;
    background: {SHEET}; font-family: "{MONO}"; font-size: 13px; selection-background-color: {INK};
}}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; height: 0; border: 0; }}
QComboBox, QLineEdit {{
    min-height: 34px; padding: 0 10px; border: 1px solid {LINE_STRONG}; border-radius: 0;
    background: {SHEET}; font-family: "{MONO}"; font-size: 13px; selection-background-color: {INK};
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {INK}; }}
QComboBox::drop-down {{ border: 0; width: 22px; }}
QComboBox QAbstractItemView {{ background: {SHEET}; border: 1px solid {INK}; selection-background-color: {PAPER_ALT}; selection-color: {INK}; }}
QCheckBox {{ spacing: 10px; min-height: 30px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1.5px solid {INK}; background: {SHEET}; }}
QCheckBox::indicator:checked {{ background: {INK}; }}

QTabWidget::pane {{ border: 0; border-top: 1px solid {HAIRLINE}; }}
QTabBar::tab {{
    background: transparent; border: 0; border-bottom: 2px solid transparent; padding: 10px 16px;
    font-family: "{MONO}"; font-size: 13px; font-weight: 600; color: {INK_SOFT};
}}
QTabBar::tab:selected {{ color: {INK}; border-bottom: 2px solid {RED}; }}

QListWidget, QTableWidget {{ background: transparent; border: 0; outline: 0; font-family: "{MONO}"; font-size: 13px; }}
QListWidget::item {{ padding: 0 12px; min-height: 30px; border-bottom: 1px solid {HAIRLINE}; }}
QListWidget::item:selected {{ background: {SHEET}; color: {INK}; }}
QTableWidget {{ gridline-color: {HAIRLINE}; }}
QTableWidget::item {{ padding: 0 8px; border-bottom: 1px solid {HAIRLINE}; }}
QTableWidget::item:selected {{ background: {SHEET}; color: {INK}; }}
QHeaderView::section {{
    background: {RED_TINT}; color: {RED_TEXT}; border: 0; padding: 6px 8px;
    font-family: "{MONO}"; font-size: 12px; font-weight: 500;
}}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {LINE_STRONG}; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QProgressBar {{ border: 1px solid {LINE_STRONG}; background: {SHEET}; height: 6px; text-align: center; }}
QProgressBar::chunk {{ background: {INK}; }}
QToolTip {{ background: {INK}; color: #FFFFFF; border: 0; padding: 4px 6px; }}
"""
