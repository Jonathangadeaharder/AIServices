from __future__ import annotations

from pathlib import Path
from typing import Literal


def transcribe(
    video_path: str | Path,
    output_path: str | Path,
    *,
    language: str | None = None,
    output_format: Literal["srt", "vtt"] = "srt",
    model_name: str = "mlx-community/whisper-large-v3-turbo",
    word_timestamps: bool = True,
    provider_name: str = "video2subtitle.mlx",
) -> Path:
    from .models import Video2SubtitleRequest
    from .providers import registry

    req = Video2SubtitleRequest(
        video_path=str(video_path),
        language=language,
        output_format=output_format,
        model_name=model_name,
        word_timestamps=word_timestamps,
    )
    provider = registry.get(provider_name)
    resp = provider.generate(req, output_path=str(Path(output_path)))
    return Path(resp.output_path)
