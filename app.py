__version__ = "2.1.0"

from ui_main_window import SpecsGraderMainWindow

def main(preloaded=None):
    """
    Launches the main DualSpecClassifierApp window.

    Args:
        preloaded: dict of models and resources loaded by the splash screen (optional).
    """
    # This function is called from the splash screen, after QApplication is created.
    # DO NOT create QApplication(sys.argv) here!
    if preloaded is not None:
        window = SpecsGraderMainWindow(preloaded=preloaded)
    else:
        window = SpecsGraderMainWindow()
    window.show()
    # Note: QApplication.exec() is called in splash.py, not here.

# If you want to support running this file directly (not required if always using splash.py):
if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = SpecsGraderMainWindow()
    window.show()
    sys.exit(app.exec())
