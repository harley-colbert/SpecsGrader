__version__ = "2.0.0"

from ui import DualSpecClassifierApp

def main(preloaded=None):
    """
    Launches the main DualSpecClassifierApp window.

    Args:
        preloaded: dict of models and resources loaded by the splash screen (optional).
    """
    # This function is called from the splash screen, after QApplication is created.
    # DO NOT create QApplication(sys.argv) here!
    if preloaded is not None:
        window = DualSpecClassifierApp(preloaded=preloaded)
    else:
        window = DualSpecClassifierApp()
    window.show()
    # Note: QApplication.exec() is called in splash.py, not here.

# If you want to support running this file directly (not required if always using splash.py):
if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = DualSpecClassifierApp()
    window.show()
    sys.exit(app.exec())
