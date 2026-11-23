"""Notes for a release: that version's section of CHANGELOG.md, then how to open an unsigned download.

    python tools/release_notes.py 0.1.0
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

OPENING = """\
## Opening the download

These downloads are not signed yet, so Windows and macOS ask before they open the app the first time.

- **Windows** (`…-windows-x64-setup.exe`): if "Windows protected your PC" appears, choose *More info*, then *Run anyway*.
- **macOS** (`…-macos-arm64.dmg` for Apple silicon, `…-macos-x64.dmg` for Intel): drag Plumbline to Applications. The first
  time, open it with a right-click and *Open*; if macOS still refuses, allow it under *System Settings › Privacy & Security*.
- **Linux** (`…-linux-x86_64.AppImage`, or the smaller `.tar.xz`): make the AppImage executable (`chmod +x`) and run it.

`SHA256SUMS.txt` lists the checksum of every file.
"""


def section(changelog: str, version: str) -> str:
    """The text under '## <version> ...' up to the next '## '. Raises ValueError when the version has no section."""
    found = re.search(rf"^## {re.escape(version)}\b[^\n]*\n(.*?)(?=^## |\Z)", changelog, re.S | re.M)
    if not found:
        raise ValueError(f"CHANGELOG.md has no section for version {version}")
    return found.group(1).strip()


def notes(version: str, root: Path | None = None) -> str:
    root = root or Path(__file__).resolve().parent.parent
    version = re.sub(r"(\.dev|rc|a|b)\d+$", "", version)      # a trial build carries the notes of the version it leads to
    return section((root / "CHANGELOG.md").read_text(encoding="utf-8"), version) + "\n\n" + OPENING


if __name__ == "__main__":
    print(notes(sys.argv[1]))
