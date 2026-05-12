from __future__ import annotations

from pathlib import Path
from typing import Literal


def extract(
    video_path: str | Path,
    output_path: str | Path,
    *,
    output_format: Literal["wav", "mp3", "aac"] = "wav",
    sample_rate: int = 44100,
    mono: bool = True,
    provider_name: str = "video2audio.ffmpeg",
) -> Path:
    from .models import Video2AudioRequest
    from .providers import registry

    req = Video2AudioRequest(
        video_path=str(video_path),
        output_format=output_format,
        sample_rate=sample_rate,
        mono=mono,
    )
    provider = registry.get(provider_name)
    resp = provider.generate(req, output_path=str(Path(output_path)))
    return Path(resp.output_path)
