"""Rebuild src/plumbline/assets/boards.json from the Calib.Tools website catalog.

The website repository is the one place where stock targets are defined
(src/lib/catalogPatterns.ts). This script reads that file through the site's own
code, so nothing is typed in by hand here.

    python tools/update_board_presets.py /path/to/calibtools-site

Needs Node.js with `tsx` on the PATH.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from plumbline.core.targets import ASYMMETRIC_CIRCLES, CHARUCO, CHECKERBOARD, CIRCLES, Target  # noqa: E402

KIND = {"ChArUco": CHARUCO, "Checkerboard": CHECKERBOARD,
        "Symmetric circles": CIRCLES, "Asymmetric circles": ASYMMETRIC_CIRCLES}

DUMP = """
import { CATALOG } from '%s/src/lib/catalogPatterns';
const types = %s;
const keep = CATALOG.filter(e => types.includes(e.spec.type));
console.log(JSON.stringify(keep.map(e => ({ pn: e.partNumber, handle: e.handle, ...e.spec }))));
"""


def main() -> int:
    site = Path(sys.argv[1]).resolve()
    sha = subprocess.run(["git", "-C", str(site), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    with tempfile.NamedTemporaryFile("w", suffix=".ts", delete=False) as fh:
        fh.write(DUMP % (site, json.dumps(sorted(KIND))))
    rows = json.loads(subprocess.run(["tsx", fh.name], capture_output=True, text=True, check=True).stdout)

    boards, dropped = [], []
    for r in rows:
        # The site counts squares: rows down, cols across (see exportConfig.ts, opencv_board_size).
        entry = {
            "part_number": r["pn"],
            "product": r["handle"],
            "kind": KIND[r["type"]],
            "squares_x": r["cols"],
            "squares_y": r["rows"],
            "square_mm": r["square"],
        }
        if entry["kind"] == CHARUCO:
            entry.update(marker_mm=r["marker"], dictionary=r["dict"], first_marker_id=r.get("startId", 0))
        elif entry["kind"] in (CIRCLES, ASYMMETRIC_CIRCLES):
            # The site walks a square lattice of rows x cols with spacing `square`; so does Target.
            if r.get("circleSpacingX") or r.get("circleSpacingY"):
                dropped.append((r["pn"], "unequal spacing across and down is not supported"))
                continue
            if r.get("circleDiameter"):
                entry.update(marker_mm=r["circleDiameter"])
        try:  # the same checks the app applies; a board the dictionary cannot hold is left out
            Target(**{k: v for k, v in entry.items() if k != "product"})
        except ValueError as exc:
            dropped.append((r["pn"], str(exc)))
            continue
        boards.append(entry)

    out = ROOT / "src" / "plumbline" / "assets" / "boards.json"
    out.write_text(json.dumps({"source": f"calibtools-site@{sha}, src/lib/catalogPatterns.ts",
                               "boards": boards}, indent=1) + "\n", encoding="utf-8")
    print(f"{len(boards)} boards written to {out.relative_to(ROOT)} from calibtools-site@{sha}")
    for pn, why in dropped:
        print(f"left out {pn}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
