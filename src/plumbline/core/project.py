"""Project files: everything needed to reopen a calibration session."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .calibrate import ModelOptions
from .detect import Detection
from .targets import Target

from datetime import date


def due_date(calibrated_on: str | None, months: int) -> str | None:
    """Calendar date `months` after `calibrated_on` (ISO), clamped to the end of shorter months."""
    if not calibrated_on or months <= 0:
        return None
    d = date.fromisoformat(calibrated_on)
    index = d.month - 1 + months
    year, month = d.year + index // 12, index % 12 + 1
    for day in (d.day, 30, 29, 28):
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            continue
    return None


def is_due(due: str | None, today: date | None = None) -> bool:
    return bool(due) and date.fromisoformat(due) <= (today or date.today())


FORMAT = "plumbline-project/1"
SUFFIX = ".plumbline"


@dataclass
class Project:
    target: Target | None = None
    options: ModelOptions = field(default_factory=ModelOptions)
    detections: list[Detection] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    excluded: set[int] = field(default_factory=set)
    path: str | None = None
    last_rms_point: float | None = None
    title: str | None = None  # the name the user sees; file names are built from it
    revision: int = 0  # number of the last export, 0 = never exported
    export_signature: str | None = None  # fingerprint of the result of the last export
    exported_on: str | None = None  # ISO date of the last export, taken as the calibration date
    recheck_months: int = 0  # remind to calibrate again after this many months, 0 = no reminder

    def due_date(self) -> str | None:
        """ISO date from which this camera should be calibrated again, or None."""
        return due_date(self.exported_on, self.recheck_months)

    @property
    def name(self) -> str:
        if self.title:
            return self.title
        return Path(self.path).stem if self.path else "Untitled"

    def revision_for(self, signature: str) -> int:
        """Revision an export of a result with this fingerprint gets. Same result, same revision."""
        if self.revision and signature == self.export_signature:
            return self.revision
        return self.revision + 1

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "target": self.target.to_dict() if self.target else None,
            "options": self.options.to_dict(),
            "image_paths": list(self.image_paths),
            "detections": [d.to_dict() for d in self.detections],
            "excluded": sorted(self.excluded),
            "last_rms_point": self.last_rms_point,
            "title": self.title,
            "revision": self.revision,
            "export_signature": self.export_signature,
            "exported_on": self.exported_on,
            "recheck_months": self.recheck_months,
        }

    def save(self, path: str | None = None) -> None:
        path = path or self.path
        if not path:
            raise ValueError("no path to save to")
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")
        self.path = str(path)

    @staticmethod
    def load(path: str) -> "Project":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError("This file is not a Plumbline project, or was made by a newer version.")
        return Project(
            target=Target.from_dict(data["target"]) if data.get("target") else None,
            options=ModelOptions.from_dict(data.get("options", {})),
            image_paths=list(data.get("image_paths", [])),
            detections=[Detection.from_dict(d) for d in data.get("detections", [])],
            excluded=set(data.get("excluded", [])),
            path=str(path),
            last_rms_point=data.get("last_rms_point"),
            title=data.get("title"),
            revision=int(data.get("revision") or 0),
            export_signature=data.get("export_signature"),
            exported_on=data.get("exported_on"),
            recheck_months=int(data.get("recheck_months") or 0),
        )
