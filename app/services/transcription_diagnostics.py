"""Confidence details of a transcription response, condensed for the log and the bad-recordings sidecar.

Whisper ``verbose_json`` (Groq, whisper-1) has log-probabilities per segment and timestamps per
word; OpenAI's GPT transcription models return a log-probability per token, which is grouped into
words here. Log-probabilities are natural logs: 0 means certain, more negative means less sure.
No Qt here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def extract(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """``{"summary", "segments", "words"}`` of a response, or None when it carries no details."""
    if isinstance(payload.get("segments"), list):
        return _from_verbose(payload)
    if isinstance(payload.get("logprobs"), list):
        return _from_logprobs(payload["logprobs"])
    return None


def _from_verbose(payload: Dict[str, Any]) -> Dict[str, Any]:
    segments = [{
        "start": s.get("start"), "end": s.get("end"), "text": s.get("text", ""),
        "avg_logprob": s.get("avg_logprob"), "no_speech_prob": s.get("no_speech_prob"),
        "compression_ratio": s.get("compression_ratio"), "temperature": s.get("temperature"),
    } for s in payload["segments"]]
    words = [{"word": w.get("word", ""), "start": w.get("start"), "end": w.get("end")}
             for w in payload.get("words") or []]
    weighted = [(s["avg_logprob"], max(0.0, (s["end"] or 0) - (s["start"] or 0))) for s in segments
                if isinstance(s["avg_logprob"], (int, float))]
    total = sum(weight for _lp, weight in weighted)
    summary: Dict[str, Any] = {
        "source": "verbose_json",
        "duration": payload.get("duration"),
        "segments": len(segments),
        "words": len(words) if words else len(payload.get("text", "").split()),
        # Weighted by segment duration, so a short garbled segment does not hide in a long good one.
        "avg_logprob": _round(sum(lp * w for lp, w in weighted) / total if total
                              else _mean([lp for lp, _w in weighted])),
        "min_segment_logprob": _round(min((lp for lp, _w in weighted), default=None)),
        "max_no_speech_prob": _round(max((s["no_speech_prob"] for s in segments
                                          if isinstance(s["no_speech_prob"], (int, float))), default=None)),
        "max_compression_ratio": _round(max((s["compression_ratio"] for s in segments
                                             if isinstance(s["compression_ratio"], (int, float))), default=None)),
    }
    return {"summary": summary, "segments": segments, "words": words}


def _from_logprobs(tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group tokens into words: a token starting with whitespace begins a new word."""
    words: List[Dict[str, Any]] = []
    for token in tokens:
        text, logprob = str(token.get("token", "")), token.get("logprob")
        if not isinstance(logprob, (int, float)):
            continue
        logprob = min(0.0, float(logprob))  # The API reports tiny positive rounding errors.
        if not words or text[:1].isspace():
            words.append({"word": text.strip(), "logprob": logprob, "min_token_logprob": logprob, "tokens": 1})
        else:
            word = words[-1]
            word["word"] += text
            word["logprob"] += logprob  # Sum of token log-probabilities = log-probability of the word.
            word["min_token_logprob"] = min(word["min_token_logprob"], logprob)
            word["tokens"] += 1
    for word in words:
        word["logprob"] = _round(word["logprob"])
        word["min_token_logprob"] = _round(word["min_token_logprob"])
    token_lps = [min(0.0, float(t["logprob"])) for t in tokens if isinstance(t.get("logprob"), (int, float))]
    summary = {
        "source": "logprobs",
        "tokens": len(token_lps),
        "words": len(words),
        "avg_logprob": _round(_mean(token_lps)),
        "min_token_logprob": _round(min(token_lps, default=None)),
        "min_word_logprob": _round(min((w["logprob"] for w in words), default=None)),
    }
    return {"summary": summary, "segments": [], "words": words}


def summary_line(details: Dict[str, Any]) -> str:
    """``key=value`` pairs of the summary for one log line (no transcript text)."""
    return " ".join(f"{key}={value}" for key, value in details["summary"].items())


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 4)
