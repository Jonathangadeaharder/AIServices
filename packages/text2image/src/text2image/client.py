from __future__ import annotations

from pathlib import Path


def _div8(n: int) -> int:
    return max(512, (n // 8) * 8)


def generate(
    prompt: str,
    output_path: str | Path,
    *,
    seed: int | None = None,
    width: int = 1024,
    height: int = 720,
    negative_prompt: str | None = None,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    provider_name: str = "text2image.mlx",
) -> Path:
    from .models import Text2ImageRequest
    from .providers import registry  # noqa: F811 – triggers registration

    req = Text2ImageRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        width=_div8(width),
        height=_div8(height),
    )
    provider = registry.get(provider_name)
    resp = provider.generate(req, output_path=str(Path(output_path)))
    return Path(resp.output_path)
