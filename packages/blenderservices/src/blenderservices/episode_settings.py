"""Episode render settings loaded from episodes/epNN/episode.toml."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blenderservices.shot_list import ShotListDocument

DEFAULT_RENDER_FPS = 24
DEFAULT_RENDER_WIDTH = 2304
DEFAULT_RENDER_HEIGHT = 960


@dataclass(frozen=True)
class EpisodeRenderSettings:
    fps: int = DEFAULT_RENDER_FPS
    width: int = DEFAULT_RENDER_WIDTH
    height: int = DEFAULT_RENDER_HEIGHT


_RESOLUTION_RE = re.compile(r"resolution\s*=\s*\[(\d+)\s*,\s*(\d+)\]")
_FPS_RE = re.compile(r"fps_source\s*=\s*(\d+)")


def load_episode_render_settings(episode_dir: Path) -> EpisodeRenderSettings:
    """Parse [render] settings from episode.toml."""
    toml_path = episode_dir / "episode.toml"
    fps = DEFAULT_RENDER_FPS
    width = DEFAULT_RENDER_WIDTH
    height = DEFAULT_RENDER_HEIGHT
    if toml_path.is_file():
        text = toml_path.read_text(encoding="utf-8")
        if match := _FPS_RE.search(text):
            fps = int(match.group(1))
        if match := _RESOLUTION_RE.search(text):
            width = int(match.group(1))
            height = int(match.group(2))
    return EpisodeRenderSettings(fps=fps, width=width, height=height)


def episode_dir_for_doc(project_root: Path, doc: ShotListDocument) -> Path:
    return project_root / "shot_production" / doc.episode


def _episode_dir_for_chapter(project_root: Path, chapter_num: int) -> Path | None:
    ch = f"{chapter_num:02d}"
    episodes = project_root / "shot_production"
    if not episodes.is_dir():
        return None
    matches = sorted(
        path for path in episodes.iterdir() if path.is_dir() and path.name.startswith(f"ep{ch}")
    )
    return matches[0] if matches else None


def load_chapter_render_settings(project_root: Path, chapter_num: int) -> EpisodeRenderSettings:
    episode_dir = _episode_dir_for_chapter(project_root, chapter_num)
    if episode_dir is None:
        return EpisodeRenderSettings()
    return load_episode_render_settings(episode_dir)
