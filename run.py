"""WhisperTyper entry point — a thin shim over app/bootstrap.py.

Bootstrap concerns (base logging, windowed crash-log redirect, project-venv re-exec,
single-instance lock + ``-r`` restart) live in app/bootstrap.py; the application itself
lives in app/application.py. Keep the stream redirect first so import-time crashes are
captured.
"""
import argparse
import sys

from app.bootstrap import (
    acquire_single_instance_lock,
    configure_base_logging,
    maybe_reexec_with_project_venv,
    redirect_std_streams_if_windowed,
)

redirect_std_streams_if_windowed()
configure_base_logging()

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox

    from app.application import WhisperTyperApp
except ModuleNotFoundError as exc:
    if exc.name == "PyQt6":
        maybe_reexec_with_project_venv()
    raise


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "-r",
        "--restart",
        action="store_true",
        help="If WhisperTyper is already running, hard-stop the existing instance and start a new one.",
    )
    return parser


def main() -> int:
    """Runs the WhisperTyper application."""
    args = _build_arg_parser().parse_args()

    app = QApplication([sys.argv[0]])
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("WhisperTyper")

    instance_lock, message = acquire_single_instance_lock(args.restart)
    if instance_lock is None:
        print(message, file=sys.stderr)
        QMessageBox.information(None, "WhisperTyper", message)
        return 0

    # Keep the lock alive for the lifetime of the Qt application.
    app._instance_lock = instance_lock
    WhisperTyperApp()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
