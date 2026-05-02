from __future__ import annotations

from pathlib import Path


def generate(
    audio_path: str | Path,
    output_path: str | Path,
    *,
    language: str | None = None,
    output_format: str = "srt",
    provider_name: str = "audio2subtitle.mlx",
) -> Path:
    from .models import Audio2SubtitleRequest
    from .providers import registry  # noqa: F811 – triggers registration

    req = Audio2SubtitleRequest(
        audio_path=str(audio_path),
        language=language,
        output_format=output_format,
    )
    provider = registry.get(provider_name)
    resp = provider.generate(req, output_path=str(Path(output_path)))
    return Path(resp.output_path if hasattr(resp, "output_path") else resp)
