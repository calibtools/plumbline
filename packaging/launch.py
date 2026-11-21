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
        print("SMOKE_OK", win.width(), win.height())
        sys.exit(0)
    win.show()
    sys.exit(app.exec())
