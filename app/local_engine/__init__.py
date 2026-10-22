"""Local transcription engine: Whisper on this computer, behind an OpenAI-compatible HTTP endpoint.

The engine runs as a child process (``python -m app.local_engine.server``) supervised by
``supervisor.Supervisor``. The modules the app imports (catalog, availability, supervisor) need
neither Qt nor the optional engine packages; those are imported by the child process only.
"""
