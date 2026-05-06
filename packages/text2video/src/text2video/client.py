from __future__ import annotations

from pathlib import Path


def _div8(n: int) -> int:
    return max(64, (n // 8) * 8)


def _frames_for_seconds(seconds: int, fps: int) -> int:
    raw = seconds * fps
    return ((raw + 7) // 8) * 8 + 1


def generate(
    prompt: str,
    output_path: str | Path,
    *,
    seed: int | None = None,
    width: int = 704,
    height: int = 480,
    seconds: int = 4,
    fps: int = 24,
    num_inference_steps: int = 8,
    negative_prompt: str | None = None,
    provider_name: str = "text2video.mlx",
) -> Path:
    from .models import Text2VideoRequest
    from .providers import registry  # noqa: F811 – triggers registration

    num_frames = _frames_for_seconds(seconds, fps)

    kwargs: dict = {
        "prompt": prompt,
        "width": _div8(width),
        "height": _div8(height),
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "fps": fps,
    }
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt

    req = Text2VideoRequest(**kwargs)
    provider = registry.get(provider_name)
    resp = provider.generate(req, output_path=str(Path(output_path)))
    return Path(resp.output_path)
