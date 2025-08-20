import sys
from PyQt6.QtWidgets import QApplication
from app.WhisterTyper import WhisperTyperApp

if __name__ == '__main__':
    """Runs the WhisperTyper application."""
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    transcriber_app = WhisperTyperApp()
    sys.exit(app.exec())
