"""Save, path resolution, and library-path normalisation."""

from __future__ import annotations

import os

import bpy


def project_root() -> str:
    """Project root: walk upward from this file until a sentinel is found."""
    cur = os.path.dirname(os.path.abspath(__file__))
    _SENTINELS = (".project.md", "project.forge.yaml", ".git", "pyproject.toml")
    for _ in range(10):
        for s in _SENTINELS:
            if os.path.exists(os.path.join(cur, s)):
                return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return cur


def resolve_blend_path(blend_path: str) -> str:
    """Resolve a repo-relative or absolute .blend path for loading."""
    if os.path.isabs(blend_path) and os.path.isfile(blend_path):
        return blend_path
    if os.path.isfile(blend_path):
        return os.path.abspath(blend_path)
    from_repo = os.path.join(project_root(), blend_path)
    if os.path.isfile(from_repo):
        return os.path.abspath(from_repo)
    return os.path.abspath(blend_path)


def normalize_library_paths(output_path: str) -> None:
    """Rewrite linked library paths relative to the file being saved."""
    blend_dir = os.path.dirname(os.path.abspath(output_path))
    asset_root = os.path.join(project_root(), "assets_3d")
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
