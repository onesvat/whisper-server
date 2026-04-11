"""Compatibility entrypoint for the old package name."""

from whisper_server.__main__ import main, run

__all__ = ["main", "run"]
