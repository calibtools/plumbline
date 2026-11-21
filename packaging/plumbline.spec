# PyInstaller spec. Build with:  pyinstaller packaging/plumbline.spec --noconfirm
import sys
from pathlib import Path

root = Path(SPECPATH).parent
src = root / "src"

a = Analysis(
    [str(root / "packaging" / "launch.py")],
    pathex=[str(src)],
    datas=[(str(src / "plumbline" / "assets"), "plumbline/assets")],
    hiddenimports=["PySide6.QtSvg"],
    excludes=["pkg_resources", "tkinter", "matplotlib", "scipy", "PySide6.QtWebEngineCore",
              "PySide6.QtQml", "PySide6.QtQuick", "PySide6.Qt3DCore", "PySide6.QtMultimedia"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Plumbline", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="Plumbline")
if sys.platform == "darwin":
    app = BUNDLE(coll, name="Plumbline.app", bundle_identifier="tools.calib.plumbline")
