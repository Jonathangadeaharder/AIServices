from __future__ import annotations

from pathlib import Path


def generate(
    image_path: str | Path,
    prompt: str,
    output_path: str | Path,
    *,
    seed: int | None = None,
    strength: float = 0.5,
    negative_prompt: str | None = None,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    provider_name: str = "image2image.mlx",
) -> Path:
    from .models import Image2ImageRequest
    from .providers import registry  # noqa: F811 – triggers registration

    req = Image2ImageRequest(
        image_path=str(image_path),
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    provider = registry.get(provider_name)
    resp = provider.generate(req, output_path=str(Path(output_path)))
    return Path(resp.output_path)
