"""Benchmark: transcribe one audio file with every downloaded model of a device, one after another.

Runs as its own process, so a running engine keeps serving:
    python -m app.local_engine.bench <audio> --device cuda --models-dir <dir> [--runs 3] [--language de]
Prints one JSON object per line: {"event": "start" | "skipped" | "loading" | "result" | "error" | "done", ...}
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from app.local_engine.catalog import MODELS, EngineSettings, backend_of, model_downloaded, model_path


def emit(**event: object) -> None:
    """Print one event as a JSON line."""
    print(json.dumps(event, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark; see the module docstring."""
    parser = argparse.ArgumentParser(prog="app.local_engine.bench")
    parser.add_argument("audio")
    parser.add_argument("--device", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--language", default="")
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from app.local_engine.server import Engine
    from app.local_engine.speech import SAMPLE_RATE, decode_audio

    data = Path(args.audio).read_bytes()
    audio_s = len(decode_audio(data)) / SAMPLE_RATE
    backend = backend_of(args.device)
    repos = args.models or [model.repo for model in MODELS[backend]]
    names = {model.repo: model.name for model in MODELS[backend]}
    base = EngineSettings(device=args.device, model=repos[0], models_dir=args.models_dir,
                          log_dir=os.path.join(tempfile.gettempdir(), "whispertyper-bench"))
    todo = []
    for repo in repos:
        if model_downloaded(backend, model_path(Path(args.models_dir), repo)):
            todo.append(dataclasses.replace(base, model=repo))
        else:
            emit(event="skipped", model=repo, name=names.get(repo, repo), reason="not downloaded")
    emit(event="start", audio_s=round(audio_s, 2), device=args.device, models=len(todo))

    for settings in todo:
        name = names.get(settings.model, settings.model)
        emit(event="loading", model=settings.model, name=name)
        engine = Engine(settings)
        t0 = time.perf_counter()
        engine.load()
        load_s = time.perf_counter() - t0
        if engine.status != "ready":
            emit(event="error", model=settings.model, name=name, error=engine.error)
            continue
        times, text = [], ""
        try:
            for _ in range(max(1, args.runs)):
                t1 = time.perf_counter()
                text, _duration = engine.transcribe(data, args.language)
                times.append(time.perf_counter() - t1)
        except Exception as exc:  # noqa: BLE001
            emit(event="error", model=settings.model, name=name, error=f"{type(exc).__name__}: {exc}")
            continue
        finally:
            engine.backend = None
            del engine
            gc.collect()
        emit(event="result", model=settings.model, name=name, load_s=round(load_s, 2), first_s=round(times[0], 2),
             best_s=round(min(times), 2), rtf=round(min(times) / audio_s, 3) if audio_s else None, text=text)
    emit(event="done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
