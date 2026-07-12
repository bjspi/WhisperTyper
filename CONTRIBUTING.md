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
mypy app/core app/services app/audio app/hotkeys app/ui   # types (Qt-free layers are strict)
pytest                                           # 100+ headless tests
```

## Ground rules

- **Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first.** The layering rules
  (pure `app/core/`, self-contained workers, queued signals for anything cross-thread)
  are what keep this codebase safe to change.
- New pure logic goes into `app/core/` **with tests** — not into a mixin.
- Never touch a Qt object from a non-main thread; add a queued signal on
  `WhisperTyperApp` instead.
- Workers get value snapshots at construction, never live config references.
- Commit style: [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat(...)`, `fix(...)`, `refactor(...)`, …) — see `git log` for examples.

## Adding a UI language

1. Copy `app/lang/en.json` to `app/lang/<code>.json` and translate the values.
2. Add the language to the selector in `app/mixins/settings_mixin.py`
   (`ui_language_selector` + `lang_map` in `change_language`).
3. `pytest tests/test_i18n_and_prompts.py` verifies your file stays key-complete.
