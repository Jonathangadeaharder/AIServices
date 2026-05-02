from __future__ import annotations

from pathlib import Path


def _div8(n: int) -> int:
    return max(64, (n // 8) * 8)


def generate(
    image_path: str | Path,
    prompt: str,
    output_path: str | Path,
    *,
    seed: int | None = None,
    width: int = 640,
    height: int = 640,
    num_frames: int = 81,
    num_inference_steps: int = 4,
    fps: int = 16,
    provider_name: str = "image2video.mlx",
) -> Path:
    from .models import Image2VideoRequest
    from .providers import registry  # noqa: F811 – triggers registration

    req = Image2VideoRequest(
        image_path=str(image_path),
        prompt=prompt,
        width=_div8(width),
        height=_div8(height),
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        seed=seed,
        fps=fps,
    )
    provider = registry.get(provider_name)
    resp = provider.generate(req, output_path=str(Path(output_path)))
    return Path(resp.output_path)
