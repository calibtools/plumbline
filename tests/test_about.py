"""The About window: what a problem report carries, and that both pages lead to it."""
import os
from urllib.parse import unquote

import pytest

import plumbline


def test_details_name_the_program_and_the_computer_only():
    pytest.importorskip("PySide6")
    from plumbline.ui.about import details, report_url

    text = details()
    lines = text.splitlines()
    assert lines[0] == f"Plumbline {plumbline.VERSION_NAME}" and len(lines) == 6
    assert any(line.startswith("OpenCV ") for line in lines) and any(line.startswith("Qt ") for line in lines)
    # Nothing about the user: no home folder, no user name, no host name.
    home = os.path.expanduser("~")
    for private in (home, os.environ.get("USER") or "\0", os.environ.get("USERNAME") or "\0"):
        assert len(private) < 3 or private not in text
    url = report_url()
    assert url.startswith("mailto:hello@calib.tools?subject=Plumbline%20")
    body = unquote(url.split("&body=")[1])
    assert text in body and "What happened instead" in body


def test_both_pages_open_the_about_window():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings, QStandardPaths, QTimer
    from PySide6.QtWidgets import QApplication

    from plumbline.app import MainWindow, create_app
    from plumbline.ui.about import AboutDialog

    QStandardPaths.setTestModeEnabled(True)
    app = create_app([])
    QSettings().clear()
    win = MainWindow()
    win.show()
    seen = []

    def look():
        for w in QApplication.topLevelWidgets():
            if isinstance(w, AboutDialog) and w.isVisible():
                w._copy()
                seen.append((w.text.text(), w.copy.text(), app.clipboard().text()))
                w.accept()

    for page in (win.welcome, win.bench):
        assert page.about_link.text() == "About"
        QTimer.singleShot(50, look)
        page.about_link.clicked.emit()
    assert len(seen) == 2 and all(shown.startswith("Plumbline ") and said == "Copied" for shown, said, _c in seen)
    assert seen[0][2] == seen[0][0] or seen[0][2] == ""      # some test displays have no clipboard


def test_release_notes_come_from_the_changelog():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "tools"))
    import release_notes

    # The version in the code has a section; a tag "v<version>" will find its notes.
    base = plumbline.VERSION_FILE.split("-")[0]
    text = release_notes.notes(base, root)
    assert release_notes.notes(plumbline.__version__, root) == text == release_notes.notes(plumbline.VERSION_FILE, root)
    assert release_notes.notes(base + "rc1", root) == text == release_notes.notes(base + "-rc.2", root)
    assert "First public version" in text and "Opening the download" in text and "## " + base not in text
    assert release_notes.section("## 0.2.0\nnew\n\n## 0.1.0\nold\n", "0.2.0") == "new"
    with pytest.raises(ValueError, match="no section"):
        release_notes.section("## 0.1.0\nold\n", "9.9.9")


def test_the_version_reads_the_same_everywhere():
    """One version, written three ways: for Python packaging, for people, for file names."""
    import re
    from pathlib import Path

    from plumbline import VERSION_FILE, VERSION_NAME, __version__, version_names

    assert version_names("0.1.0b1") == ("0.1.0 beta 1", "0.1.0-beta.1")
    assert version_names("0.1.0rc2") == ("0.1.0 rc 2", "0.1.0-rc.2")
    assert version_names("1.2.0") == ("1.2.0", "1.2.0")
    assert version_names("0.2.0.dev3") == ("0.2.0 dev 3", "0.2.0-dev.3")
    with pytest.raises(ValueError):
        version_names("next")
    assert (VERSION_NAME, VERSION_FILE) == version_names(__version__)
    root = Path(__file__).resolve().parent.parent
    declared = re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.M).group(1)
    assert declared == __version__
    # File names carry no spaces and nothing a shell or a web address would trip over.
    assert re.fullmatch(r"[0-9A-Za-z.-]+", VERSION_FILE)
    # The downloads and the tag are named from the file form.
    workflow = (root / ".github" / "workflows" / "package.yml").read_text()
    assert workflow.count("steps.v.outputs.file") >= 6 and "Plumbline-${{ steps.v.outputs.file }}-${{ matrix.name }}" in workflow
