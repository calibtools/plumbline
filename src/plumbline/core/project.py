"""Project files: everything needed to reopen a calibration session."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .calibrate import ModelOptions
from .detect import Detection
from .targets import Target

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

    @property
    def name(self) -> str:
        return Path(self.path).stem if self.path else "Untitled"

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "target": self.target.to_dict() if self.target else None,
            "options": self.options.to_dict(),
            "image_paths": list(self.image_paths),
            "detections": [d.to_dict() for d in self.detections],
            "excluded": sorted(self.excluded),
            "last_rms_point": self.last_rms_point,
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
        )
