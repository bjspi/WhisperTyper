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
    enable_native_crash_log,
    install_qt_message_handler,
    mark_clean_exit,
    maybe_reexec_with_project_venv,
    redirect_std_streams_if_windowed,
)

redirect_std_streams_if_windowed()
configure_base_logging()

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox

    from app.application import WhisperTyperApp
    from app.ui import tooltip_wrap
    from app.ui.fonts import install_emoji_font
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
    parser.add_argument(
        "--show-settings",
        nargs="?",
        type=int,
        const=0,
        default=None,
        metavar="TAB",
        help="Open the settings window on tab TAB (default 0) after startup; a restart passes the open tab.",
    )
    return parser


def main() -> int:
    """Runs the WhisperTyper application."""
    args = _build_arg_parser().parse_args()

    app = QApplication([sys.argv[0]])
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("WhisperTyper")
    install_emoji_font()
    tooltip_wrap.install(app)

    instance_lock, message = acquire_single_instance_lock(args.restart)
    if instance_lock is None:
        print(message, file=sys.stderr)
        QMessageBox.information(None, "WhisperTyper", message)
        return 0
    # Only the instance that keeps running records crashes and Qt's own warnings.
    install_qt_message_handler()
    enable_native_crash_log()

    # The instance lock and the application stay referenced while the event loop runs.
    application = WhisperTyperApp(show_settings=args.show_settings)
    exit_code = app.exec()
    mark_clean_exit()
    del application, instance_lock
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
