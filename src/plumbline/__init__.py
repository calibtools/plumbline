"""Plumbline: camera calibration software by Calib.Tools."""
import re

# The one place the version is set. `pyproject.toml` carries the same string (a test keeps them together).
# Written the way Python packaging wants it: 0.1.0b1 is the first beta of 0.1.0, 0.1.0rc1 a release
# candidate, 0.1.0 the release itself.
__version__ = "0.1.0b1"

_STAGES = {"a": "alpha", "b": "beta", "rc": "rc", ".dev": "dev"}


def version_names(version: str) -> tuple[str, str]:
    """How a version reads to people and in file names: '0.1.0b1' -> ('0.1.0 beta 1', '0.1.0-beta.1')."""
    found = re.fullmatch(r"(\d+(?:\.\d+)*)(?:(a|b|rc|\.dev)(\d+))?", version)
    if not found:
        raise ValueError(f"not a version Plumbline can name: {version}")
    number, stage, count = found.groups()
    if stage is None:
        return number, number
    word = _STAGES[stage]
    return f"{number} {word} {count}", f"{number}-{word}.{count}"


VERSION_NAME, VERSION_FILE = version_names(__version__)   # as shown in the app and in what it writes; as used in file names
