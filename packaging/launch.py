import os
import sys

from plumbline.app import MainWindow, create_app

if __name__ == "__main__":
    app = create_app()
    win = MainWindow()
    if os.environ.get("PLUMBLINE_SMOKE"):
        # Used by the build check: open the window off screen, then leave.
        win.show()
        app.processEvents()
        line = f"SMOKE_OK {win.width()} {win.height()}"
        print(line)
        if os.environ.get("PLUMBLINE_SMOKE_FILE"):  # windowed builds on Windows have no console
            with open(os.environ["PLUMBLINE_SMOKE_FILE"], "w", encoding="utf-8") as fh:
                fh.write(line)
        sys.exit(0)
    win.show()
    sys.exit(app.exec())
