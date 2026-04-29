# MLX FLUX Provider for Image Generation

**Date:** 2026-04-28
**Package:** `packages/image2image`
**Status:** Approved

## Summary

Add a native MLX provider to the `image2image` package that runs FLUX models directly on Apple Silicon, supporting both text-to-image and image-to-image generation. Uses Apple's `mlx-examples` FLUX implementation as the foundation.

## Motivation

The current `image2image` package relies on ComfyUI (requires a running server) or Replicate (cloud API). An MLX provider enables:
- Fully local inference on Apple Silicon with no external dependencies
- Faster iteration (no server setup, no API latency)
- Both text-to-image and image-to-image in a single provider

## Design

### Request Model Changes

Make `image_path` optional in `Image2ImageRequest` and add dimension fields:

```python
class Image2ImageRequest(BaseModel):
    image_path: str | None = Field(None, description="Path to input image (None = text-to-image)")
    prompt: str = Field(..., description="Text prompt")
    negative_prompt: str | None = None
    strength: float = Field(0.5, ge=0.0, le=1.0)
    guidance_scale: float = Field(7.5)
    num_inference_steps: int = Field(50)
    seed: int | None = None
    width: int = Field(1024, description="Output width in pixels")
    height: int = Field(1024, description="Output height in pixels")
```

When `image_path` is `None`, the provider runs text-to-image. When provided, it runs image-to-image using `strength` to control transformation intensity. This is backward-compatible — existing callers that always pass `image_path` continue to work unchanged.

### Provider Implementation

New file: `providers/mlx.py`

```python
class MLXFluxProvider(BaseProvider):
    def __init__(self, model_name: str = "flux-schnell", **kwargs):
        super().__init__(**kwargs)
        self.pipeline = FluxPipeline(model_name)

    def generate(self, request: Image2ImageRequest, output_path: str) -> Image2ImageResponse:
        if request.image_path is not None:
            return self._img2img(request, output_path)
        return self._txt2img(request, output_path)
```

**Text-to-image (`_txt2img`):**
- Uses `pipeline.generate_images()` to generate from text prompt
- Computes latent size from `width` and `height` (divided by 8, then by 2 for patchification)
- Passes `guidance_scale`, `num_inference_steps`, `seed`

**Image-to-image (`_img2img`):**
1. Load input image via PIL, resize to `(width, height)`
2. Encode to latents via `pipeline.ae.encode()`
3. Add noise using FLUX's flow-matching forward process: `x_t = x_0 * (1 - t_start) + t_start * noise` where `t_start = 1.0 - strength`
4. Run denoising loop from `t_start` to `0` using `sampler.timesteps(num_steps, seq_len, start=t_start, stop=0)`
5. Decode final latents via `pipeline.ae.decode()`
6. Save output image

The FLUX sampler's `add_noise(x, t, noise)` method handles step 3, and `timesteps(start, stop)` handles step 4. When `strength=0`, no noise is added (identity). When `strength=1`, full noise is added (equivalent to text-to-image).

### Vendored FLUX Module

Copy the `flux/flux/` directory from [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples/tree/main/flux/flux) into `packages/image2image/src/image2image/flux_mlx/`.

Files to vendor:
- `__init__.py`
- `autoencoder.py`
- `clip.py`
- `flux.py` (contains `FluxPipeline`)
- `layers.py`
- `lora.py`
- `model.py`
- `sampler.py`
- `t5.py`
- `tokenizers.py`
- `utils.py`

Extend `utils.py` to support HuggingFace model IDs like `mlx-community/FLUX.1-schnell` in addition to the built-in names ("flux-schnell", "flux-dev").

### Provider Registration

In `providers/__init__.py`:
```python
from .mlx import MLXFluxProvider
registry.register("image2image.mlx", MLXFluxProvider)
```

### Dependencies

Add to `pyproject.toml`:
```toml
dependencies = [
    # ... existing deps ...
    "mlx>=0.21.0",
    "huggingface-hub>=0.20.0",
    "sentencepiece>=0.2.0",
]
```

### CLI Changes

In `cli.py`:
- Make `--input`/`-i` optional (omit for text-to-image mode)
- Add `--width` and `--height` options
- Keep default provider as `image2image.comfyui`

## Files to Create/Modify

| File | Action |
|------|--------|
| `packages/image2image/src/image2image/models.py` | Modify — optional `image_path`, add `width`/`height` |
| `packages/image2image/src/image2image/providers/mlx.py` | Create — MLX FLUX provider |
| `packages/image2image/src/image2image/providers/__init__.py` | Modify — register new provider |
| `packages/image2image/src/image2image/flux_mlx/` | Create — vendored FLUX module (11 files) |
| `packages/image2image/pyproject.toml` | Modify — add mlx dependencies |
| `packages/image2image/src/image2image/cli.py` | Modify — optional input, add width/height |

## Model Support

Default model: `flux-schnell` (fast, 4-step generation)

Supported models via `model_name` parameter:
- `flux-schnell` — Timestep-distilled, fast (4 steps)
- `flux-dev` — Guidance-distilled, higher quality (50 steps)
- Any `mlx-community/FLUX.1-*` HuggingFace model

Models are downloaded from HuggingFace Hub on first use and cached in `~/.cache/huggingface/`.

## Error Handling

- Missing input image: raise `FileNotFoundError` (img2img mode)
- Invalid dimensions: validate `width`/`height` are divisible by 16 (FLUX requirement)
- Model download failure: propagate `huggingface_hub` errors with clear message
- Memory errors: FLUX models are large (9B+ params). Log a warning if available memory is insufficient.

## Testing

- Unit tests for `Image2ImageRequest` validation with optional `image_path`
- Integration test: generate a small image (64x64) with `flux-schnell` to verify the pipeline works
- Both t2i and i2i code paths should be tested

## Notes

- The `model_name` parameter accepts any model name supported by the vendored FLUX module. Default is `flux-schnell`. For `mlx-community/FLUX.2-klein-9B` or similar, the model loading code in `utils.py` may need to be extended to resolve the HuggingFace model ID to the correct weight files.
- FLUX models are large (9B+ parameters for the klein variant). On machines with 16GB RAM, quantization may be required. The mlx-examples FLUX implementation supports quantization via the `--quantize` flag; we should expose this as an optional `quantize` parameter on the provider.
