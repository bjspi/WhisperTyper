# Contributing to WhisperTyper

Thanks for your interest! Issues and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/bjspi/WhisperTyper.git
cd WhisperTyper
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -e ".[dev]"
python run.py
```

> **macOS**: install PortAudio first (`brew install portaudio`) — see the README for the
> full macOS setup including permissions.

## Before you open a PR

Run the same checks CI runs:

```bash
ruff check app tests run.py                      # lint
mypy app run.py                                  # types (every package is checked)
pytest                                           # 100+ headless tests
```

## Ground rules

- **Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first.** The layering rules
  (pure `app/core/`, self-contained workers, queued signals for anything cross-thread)
  are what keep this codebase safe to change.
- No mixins: a component receives its collaborators in its constructor, and
  `app/application.py` wires them together.
- New pure logic goes into `app/core/` **with tests** — not into a controller or widget.
- Never touch a Qt object from a non-main thread; emit a signal of a GUI-thread object
  (e.g. the `Notifier`, or a signal on the owning controller) instead.
- Workers get value snapshots at construction, never live config references.
- Commit style: [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat(...)`, `fix(...)`, `refactor(...)`, …) — see `git log` for examples.

## Adding a UI language

1. Copy `app/lang/en.json` to `app/lang/<code>.json` and translate the values.
2. Add the language code and its display name to `UI_LANGUAGES` in `app/core/i18n.py`.
3. `pytest tests/test_i18n_and_prompts.py` verifies your file stays key-complete.
