"""Save, path resolution, and library-path normalisation."""

from __future__ import annotations

import os

import bpy


def repo_root() -> str:
    """Repository root (parent of tools/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_blend_path(blend_path: str) -> str:
    """Resolve a repo-relative or absolute .blend path for loading."""
    if os.path.isabs(blend_path) and os.path.isfile(blend_path):
        return blend_path
    if os.path.isfile(blend_path):
        return os.path.abspath(blend_path)
    from_repo = os.path.join(repo_root(), blend_path)
    if os.path.isfile(from_repo):
        return os.path.abspath(from_repo)
    return os.path.abspath(blend_path)


def normalize_library_paths(output_path: str) -> None:
    """Rewrite linked library paths relative to the file being saved."""
    blend_dir = os.path.dirname(os.path.abspath(output_path))
    asset_root = os.path.join(repo_root(), "assets_3d")
    for lib in bpy.data.libraries:
        abs_path = bpy.path.abspath(lib.filepath, start=blend_dir)
        if not os.path.isfile(abs_path):
            basename = os.path.basename(lib.filepath)
            for root, _, files in os.walk(asset_root):
                if basename in files:
                    abs_path = os.path.join(root, basename)
                    break
        if os.path.isfile(abs_path):
            lib.filepath = bpy.path.relpath(abs_path, start=blend_dir)


def save_as(path: str) -> None:
    """Write the current file. Creates parent dirs if needed."""
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    normalize_library_paths(abs_path)
    bpy.ops.wm.save_as_mainfile(filepath=abs_path)
    print(f"✓ saved → {path}")


def log_step(step: str) -> None:
    print(f"   · {step}")
