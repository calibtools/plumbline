"""Names of projects and of the files an export writes."""
from __future__ import annotations

import hashlib
import re

import numpy as np

DEFAULT_PREFIX = "Calibration"
_FORBIDDEN = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def default_name(number: int) -> str:
    """Name of the n-th project made on this computer: Calibration-001, Calibration-002, ..."""
    return f"{DEFAULT_PREFIX}-{number:03d}"


def safe_name(text: str) -> str:
    """A project name that can be used in a file name on Windows, macOS and Linux."""
    name = _FORBIDDEN.sub("", text or "")
    name = re.sub(r"\s+", "-", name.strip())
    name = name.strip(".-")[:60].rstrip(".-")
    if not name or name.upper() in _RESERVED:
        return DEFAULT_PREFIX
    return name


def export_stem(name: str, revision: int) -> str:
    """Common start of every file of one export. Revision 1 carries no suffix."""
    stem = safe_name(name)
    return stem if revision <= 1 else f"{stem}-r{revision}"


def export_files(name: str, revision: int, camera: int | None = None) -> dict[str, str]:
    """File names of one export. With two cameras, `camera` (1 or 2) gives that camera's own set."""
    stem = export_stem(name, revision)
    if camera is not None:
        stem = f"{stem}-cam{camera}"
    return {
        "report": f"{stem}-report.pdf",
        "opencv_yaml": f"{stem}-camera.yml",
        "opencv_xml": f"{stem}-camera.xml",
        "json": f"{stem}-camera.json",
        "xlsx": f"{stem}-data.xlsx",
        "ros": f"{stem}-camera_info.yaml",
        "maps": f"{stem}-undistort-maps.npz",
        "example": f"{stem}-example.py",
        "project": f"{safe_name(name)}.plumbline",
    }


def rig_files(name: str, revision: int) -> dict[str, str]:
    """The files that say how two cameras sit to each other."""
    stem = export_stem(name, revision)
    return {"rig_json": f"{stem}-cameras.json", "rig_yaml": f"{stem}-cameras.yml",
            "rig_report": f"{stem}-cameras-report.pdf"}


def rig_signature(rig) -> str:
    """Fingerprint of a two-camera result."""
    h = hashlib.sha256()
    h.update((result_signature(rig.first) + result_signature(rig.second)).encode())
    h.update(np.round(rig.rotation, 9).tobytes())
    h.update(np.round(rig.translation, 6).tobytes())
    return h.hexdigest()[:16]


def result_signature(result) -> str:
    """Fingerprint of a result. Two exports with the same fingerprint are the same revision."""
    h = hashlib.sha256()
    h.update(np.round(result.camera_matrix, 9).tobytes())
    h.update(np.round(result.distortion, 12).tobytes())
    h.update(repr((result.image_width, result.image_height, sorted(result.excluded),
                   [v.index for v in result.views], result.options.to_dict(),
                   result.target.to_dict())).encode())
    return h.hexdigest()[:16]
