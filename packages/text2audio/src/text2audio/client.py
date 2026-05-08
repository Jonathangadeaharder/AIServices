from __future__ import annotations

from pathlib import Path
from typing import Literal


def generate(
    prompt: str,
    output_path: str | Path,
    *,
    duration_seconds: float = 10.0,
    output_format: Literal["wav", "mp3"] = "wav",
    category: str = "music",
    seed: int | None = None,
    provider_name: str = "text2audio.mlx",
) -> Path:
    from .models import AudioCategory, Text2AudioRequest
    from .providers import registry

    req = Text2AudioRequest(
        prompt=prompt,
        duration_seconds=duration_seconds,
        output_format=output_format,
        category=AudioCategory(category),
        seed=seed,
    )
    provider = registry.get(provider_name)
    resp = provider.generate(req, output_path=str(Path(output_path)))
    return Path(resp.output_path)
