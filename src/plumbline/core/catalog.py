"""Stock Calib.Tools targets, by part number.

The list is generated from the Calib.Tools website catalog by
tools/update_board_presets.py and shipped with the app; nothing here is typed by hand.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .targets import Target

BOARDS_FILE = Path(__file__).resolve().parent.parent / "assets" / "boards.json"


@dataclass(frozen=True)
class StockBoard:
    target: Target
    product: str  # page name on calib.tools, under /targets/

    @property
    def part_number(self) -> str:
        return self.target.part_number

    @property
    def product_path(self) -> str:
        return f"/targets/{self.product}/"


@lru_cache(maxsize=1)
def stock_boards() -> tuple[StockBoard, ...]:
    try:
        data = json.loads(BOARDS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    out = []
    for entry in data.get("boards", []):
        fields = {k: v for k, v in entry.items() if k != "product"}
        try:
            out.append(StockBoard(Target(**fields), entry.get("product", "")))
        except (TypeError, ValueError):
            continue  # a board this version cannot describe is skipped, not guessed at
    return tuple(out)


def find_board(part_number: str | None) -> StockBoard | None:
    if not part_number:
        return None
    return next((b for b in stock_boards() if b.part_number == part_number), None)


def same_geometry(a: Target, b: Target) -> bool:
    """True when two targets are the same pattern, whatever part number they carry."""
    def strip(t: Target) -> dict:
        # The circle diameter plays no part in detection or calibration, only in drawing.
        skip = ("part_number", "marker_mm") if t.is_circles else ("part_number",)
        return {k: v for k, v in t.to_dict().items() if k not in skip}

    return strip(a) == strip(b)
