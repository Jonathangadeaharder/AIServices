from __future__ import annotations

from pathlib import Path


def generate(
    text: str,
    output_path: str | Path,
    *,
    voice_id: str | None = None,
    emotion: str | None = None,
    tone: str | None = None,
    effect: str | None = None,
    reference_audio: str | None = None,
    provider_name: str = "text2speech.fish_mlx",
) -> Path:
    from .models import Text2SpeechRequest
    from .providers import registry  # noqa: F811 – triggers registration

    req = Text2SpeechRequest(
        text=text,
        voice_id=voice_id,
        emotion=emotion,
        tone=tone,
        effect=effect,
        reference_audio=reference_audio,
    )
    provider = registry.get(provider_name)
    resp = provider.generate(req, output_path=str(Path(output_path)))
    return Path(resp.output_path)
