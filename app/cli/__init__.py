"""The ``whispertyper`` command line: the tray app, plus transcription and rephrasing for scripts and tools.

Every command except ``run`` works without Qt and without a running instance: it reads the saved
configuration (keys, models, prompts, corrections) and sends the same requests the tray app sends.
"""
