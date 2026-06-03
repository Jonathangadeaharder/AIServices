"""Shared Blender-script utilities."""

from __future__ import annotations

import sys


def blender_argv(argv: list[str] | None = None) -> list[str]:
    """Return args after Blender's ``--`` separator."""
    raw = sys.argv if argv is None else argv
    return raw[raw.index("--") + 1 :] if "--" in raw else []
