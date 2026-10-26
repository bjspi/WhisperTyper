"""WhisperTyper entry point — a thin shim over the command line (app/cli/main.py).

Without a command it starts the tray app (app/host.py); ``python run.py --help`` lists the
commands. Keep the stream redirect first so import-time crashes of a windowless start are
captured.
"""
from app.bootstrap import maybe_reexec_with_project_venv, redirect_std_streams_if_windowed, resolve_missing_dependency

redirect_std_streams_if_windowed()

try:
    from app.cli.main import main
except ModuleNotFoundError as exc:
    # Another interpreter may have the packages (the project venv); otherwise an update added one.
    maybe_reexec_with_project_venv()
    resolve_missing_dependency(exc.name or str(exc))
    raise


if __name__ == '__main__':
    main()
