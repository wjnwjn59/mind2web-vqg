"""CLI helpers: make sure non-ASCII output works on Windows (cp1252 consoles)."""

from __future__ import annotations

import sys


def ensure_utf8() -> None:
    """Force stdout/stderr to UTF-8 to avoid UnicodeEncodeError on non-ASCII output.

    Safe to call repeatedly and when the streams are files/pipes.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass
