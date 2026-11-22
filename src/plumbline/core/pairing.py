"""Images of two cameras handed over together: tell them apart and pair them by their file names.

Typical sets: `left_001.png` / `right_001.png`, `cam1-07.png` / `cam2-07.png`,
`0007_0.png` / `0007_1.png`, or two folders `left/` and `right/` with the same
file names inside. The rule does not know these words: it looks for the one part
of the name (or the folder) that takes exactly two values, and whose removal
leaves names that match up one to one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Split:
    first: list[str]          # camera 1, in pair order
    second: list[str]         # camera 2, same order: second[i] belongs with first[i]
    labels: tuple[str, str]   # what told the cameras apart, e.g. ("left", "right")
    alone: list[str]          # images whose partner is missing; they are in neither list


def _parts(path: str) -> list[str]:
    """Folder and file name cut into pieces: runs of letters, runs of digits, and what separates them."""
    p = Path(path)
    return [p.parent.name, "/"] + re.findall(r"[A-Za-z]+|\d+|[^A-Za-z\d]+", p.stem.lower()) + [p.suffix.lower()]


def natural(pieces) -> list:
    """Sort key that puts 2 before 10. Takes a path or the pieces of one."""
    if isinstance(pieces, str):
        pieces = _parts(pieces)
    return [(0, int(x), "") if x.isdigit() else (1, 0, x) for x in pieces]


def split_by_camera(paths: list[str], least: int = 3) -> Split | None:
    """Split a set of image paths into two cameras, or None if the names do not say so clearly."""
    if len(paths) < 2 * least:
        return None
    parts = {p: _parts(p) for p in paths}
    lengths = {len(v) for v in parts.values()}
    if len(lengths) != 1:
        return None
    best = None
    for at in range(lengths.pop()):
        values = sorted({v[at] for v in parts.values()})
        if len(values) != 2:
            continue
        groups: dict[tuple, dict[str, str]] = {}
        clash = False
        for p, v in parts.items():
            key = tuple(v[:at] + v[at + 1:])
            side = groups.setdefault(key, {})
            if v[at] in side:
                clash = True      # two images of the same camera would share a pair
                break
            side[v[at]] = p
        if clash:
            continue
        pairs = sorted(((k, g) for k, g in groups.items() if len(g) == 2), key=lambda kg: natural(kg[0]))
        alone = sorted(p for g in groups.values() if len(g) == 1 for p in g.values())
        # Most images must have a partner; a part that merely happens to take two values does not pass this.
        if len(pairs) < least or len(alone) > len(pairs) // 4:
            continue
        if best is None or len(pairs) > len(best[0]):
            best = (pairs, alone, values)
    if best is None:
        return None
    pairs, alone, (a, b) = best
    return Split([g[a] for _k, g in pairs], [g[b] for _k, g in pairs], (a, b), alone)
