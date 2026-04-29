# MLX FLUX Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native MLX provider to the image2image package that runs FLUX models on Apple Silicon, supporting both text-to-image and image-to-image generation.

**Architecture:** Vendor Apple's mlx-examples FLUX pipeline into the image2image package, create a provider that wraps it with the standard BaseProvider interface, and make the request model support both t2i and i2i modes via an optional `image_path`.

**Tech Stack:** Python, mlx, mlx.nn, huggingface-hub, sentencepiece, PIL, pydantic, typer

---

## File Structure

| File | Responsibility |
|------|---------------|
| `packages/image2image/src/image2image/flux_mlx/__init__.py` | Expose FluxPipeline |
| `packages/image2image/src/image2image/flux_mlx/autoencoder.py` | VAE encoder/decoder |
| `packages/image2image/src/image2image/flux_mlx/clip.py` | CLIP text encoder |
| `packages/image2image/src/image2image/flux_mlx/flux.py` | FluxPipeline class |
| `packages/image2image/src/image2image/flux_mlx/layers.py` | Neural network layers |
| `packages/image2image/src/image2image/flux_mlx/lora.py` | LoRA support |
| `packages/image2image/src/image2image/flux_mlx/model.py` | FLUX transformer model |
| `packages/image2image/src/image2image/flux_mlx/sampler.py` | Flow-matching sampler |
| `packages/image2image/src/image2image/flux_mlx/t5.py` | T5 text encoder |
| `packages/image2image/src/image2image/flux_mlx/tokenizers.py` | Tokenizer implementations |
| `packages/image2image/src/image2image/flux_mlx/utils.py` | Model loading utilities |
| `packages/image2image/src/image2image/models.py` | Request/response models |
| `packages/image2image/src/image2image/providers/mlx.py` | MLX FLUX provider |
| `packages/image2image/src/image2image/providers/__init__.py` | Provider registration |
| `packages/image2image/src/image2image/cli.py` | CLI interface |
| `packages/image2image/pyproject.toml` | Package dependencies |
| `packages/image2image/tests/test_models.py` | Model validation tests |
| `packages/image2image/tests/test_mlx_provider.py` | Provider integration tests |

---

### Task 1: Vendor the FLUX Module from mlx-examples

**Files:**
- Create: `packages/image2image/src/image2image/flux_mlx/__init__.py`
- Create: `packages/image2image/src/image2image/flux_mlx/autoencoder.py`
- Create: `packages/image2image/src/image2image/flux_mlx/clip.py`
- Create: `packages/image2image/src/image2image/flux_mlx/flux.py`
- Create: `packages/image2image/src/image2image/flux_mlx/layers.py`
- Create: `packages/image2image/src/image2image/flux_mlx/lora.py`
- Create: `packages/image2image/src/image2image/flux_mlx/model.py`
- Create: `packages/image2image/src/image2image/flux_mlx/sampler.py`
- Create: `packages/image2image/src/image2image/flux_mlx/t5.py`
- Create: `packages/image2image/src/image2image/flux_mlx/tokenizers.py`
- Create: `packages/image2image/src/image2image/flux_mlx/utils.py`

- [ ] **Step 1: Download all FLUX source files from mlx-examples**

Download the following files from `https://github.com/ml-explore/mlx-examples/tree/main/flux/flux/` and save them to `packages/image2image/src/image2image/flux_mlx/`:
- `__init__.py`
- `autoencoder.py`
- `clip.py`
- `flux.py`
- `layers.py`
- `lora.py`
- `model.py`
- `sampler.py`
- `t5.py`
- `tokenizers.py`
- `utils.py`

- [ ] **Step 2: Update relative imports in vendored files**

The vendored files use relative imports like `from .autoencoder import ...`. These should already be correct since we're preserving the package structure. Verify each file's imports reference sibling modules within the `flux_mlx` package.

- [ ] **Step 3: Update `__init__.py` to expose FluxPipeline**

Ensure `packages/image2image/src/image2image/flux_mlx/__init__.py` exports `FluxPipeline`:

```python
from .flux import FluxPipeline

__all__ = ["FluxPipeline"]
```

- [ ] **Step 4: Verify vendored module imports**

Run from the project root:
```bash
cd /Users/jonathangadeaharder/Documents/projects/AIServices
uv run python -c "from image2image.flux_mlx import FluxPipeline; print('Import OK')"
```
Expected: Should fail with a missing dependency error (mlx not installed yet), but the import path itself should resolve. If there's an ImportError for the module itself, fix the file paths.

- [ ] **Step 5: Commit vendored FLUX module**

```bash
git add packages/image2image/src/image2image/flux_mlx/
git commit -m "feat: vendor mlx-examples FLUX module into image2image package"
```

---

### Task 2: Update Image2ImageRequest Model

**Files:**
- Modify: `packages/image2image/src/image2image/models.py`
- Create: `packages/image2image/tests/test_models.py`

- [ ] **Step 1: Write failing test for optional image_path**

Create `packages/image2image/tests/test_models.py`:

```python
from image2image.models import Image2ImageRequest


def test_request_with_image_path():
    req = Image2ImageRequest(
        image_path="/tmp/test.png",
        prompt="a cat",
    )
    assert req.image_path == "/tmp/test.png"
    assert req.width == 1024
    assert req.height == 1024


def test_request_without_image_path():
    req = Image2ImageRequest(prompt="a cat")
    assert req.image_path is None


def test_request_custom_dimensions():
    req = Image2ImageRequest(prompt="a cat", width=512, height=768)
    assert req.width == 512
    assert req.height == 768
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/image2image/tests/test_models.py -v`
Expected: FAIL — `image_path` is required, `width`/`height` fields don't exist

- [ ] **Step 3: Update Image2ImageRequest**

Modify `packages/image2image/src/image2image/models.py`:

```python
from typing import Any

from pydantic import BaseModel, Field


class Image2ImageRequest(BaseModel):
    """Request model for image generation (text-to-image and image-to-image)."""

    image_path: str | None = Field(None, description="Path or URL to the input image (None for text-to-image)")
    prompt: str = Field(..., description="Text prompt for the generation")
    negative_prompt: str | None = Field(None, description="Negative text prompt")
    strength: float = Field(0.5, ge=0.0, le=1.0, description="Transformation strength (image-to-image only)")
    guidance_scale: float = Field(7.5, description="Guidance scale (CFG)")
    num_inference_steps: int = Field(50, description="Number of denoising steps")
    seed: int | None = Field(None, description="Random seed")
    width: int = Field(1024, description="Output width in pixels")
    height: int = Field(1024, description="Output height in pixels")


class Image2ImageResponse(BaseModel):
    """Response model for image generation."""

    output_path: str = Field(..., description="Path where the output image is saved")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/image2image/tests/test_models.py -v`
Expected: 3 passed

- [ ] **Step 5: Run existing tests to verify backward compatibility**

Run: `uv run pytest packages/image2image/ -v`
Expected: All existing tests pass (the ComfyUI and Replicate providers always pass `image_path`, so they're unaffected)

- [ ] **Step 6: Commit model changes**

```bash
git add packages/image2image/src/image2image/models.py packages/image2image/tests/test_models.py
git commit -m "feat: make image_path optional, add width/height to Image2ImageRequest"
```

---

### Task 3: Create the MLX FLUX Provider

**Files:**
- Create: `packages/image2image/src/image2image/providers/mlx.py`

- [ ] **Step 1: Create the MLX provider file**

Create `packages/image2image/src/image2image/providers/mlx.py` with the complete implementation:

```python
from pathlib import Path

import mlx.core as mx
import numpy as np
from aiservices_core.providers import BaseProvider
from PIL import Image

from ..flux_mlx import FluxPipeline
from ..models import Image2ImageRequest, Image2ImageResponse


def _to_latent_size(image_size: tuple[int, int]) -> tuple[int, int]:
    """Convert pixel dimensions to latent dimensions for FLUX.

    Dimensions must be divisible by 16. Returns (h//8, w//8).
    """
    h, w = image_size
    h = ((h + 15) // 16) * 16
    w = ((w + 15) // 16) * 16
    return (h // 8, w // 8)


class MLXFluxProvider(BaseProvider):
    """Local FLUX provider using MLX (Apple Silicon).

    Supports both text-to-image and image-to-image generation.
    Uses the vendored mlx-examples FLUX implementation.
    """

    def __init__(
        self,
        model_name: str = "flux-schnell",
        t5_padding: bool = True,
        quantize: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.t5_padding = t5_padding
        self._quantize = quantize
        self._pipeline: FluxPipeline | None = None

    @property
    def pipeline(self) -> FluxPipeline:
        """Lazy-load the pipeline on first use."""
        if self._pipeline is None:
            self._pipeline = FluxPipeline(self.model_name, t5_padding=self.t5_padding)
            if self._quantize:
                import mlx.nn as nn
                from ..flux_mlx.utils import quantization_predicate

                nn.quantize(
                    self._pipeline.flow,
                    class_predicate=quantization_predicate,
                )
                nn.quantize(
                    self._pipeline.t5,
                    class_predicate=quantization_predicate,
                )
                nn.quantize(
                    self._pipeline.clip,
                    class_predicate=quantization_predicate,
                )
        return self._pipeline

    def generate(
        self, request: Image2ImageRequest, output_path: str
    ) -> Image2ImageResponse:
        if request.image_path is not None:
            return self._img2img(request, output_path)
        return self._txt2img(request, output_path)

    def _txt2img(
        self, request: Image2ImageRequest, output_path: str
    ) -> Image2ImageResponse:
        """Generate an image from a text prompt."""
        latent_size = _to_latent_size((request.height, request.width))

        images = self.pipeline.generate_images(
            text=request.prompt,
            n_images=1,
            num_steps=request.num_inference_steps,
            guidance=request.guidance_scale,
            latent_size=latent_size,
            seed=request.seed,
            progress=True,
        )

        img_array = np.array(images[0])
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

        return Image2ImageResponse(
            output_path=output_path,
            metadata={
                "provider": "mlx",
                "model": self.model_name,
                "mode": "txt2img",
                "seed": request.seed,
                "width": request.width,
                "height": request.height,
            },
        )

    def _img2img(
        self, request: Image2ImageRequest, output_path: str
    ) -> Image2ImageResponse:
        """Generate an image from an input image + text prompt."""
        input_path = Path(request.image_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {request.image_path}")

        # Load and resize input image
        input_img = Image.open(input_path).convert("RGB")
        input_img = input_img.resize((request.width, request.height))

        # Convert to array normalized to [-1, 1] for the VAE
        img_array = np.array(input_img).astype(np.float32) / 255.0
        img_array = img_array * 2.0 - 1.0
        img_array = mx.array(img_array)[None]  # Add batch dimension

        # Encode to latent space
        latent_size = _to_latent_size((request.height, request.width))
        x_0 = self.pipeline.ae.encode(img_array)

        # Reshape to patchified latent format
        x_0, x_ids = self.pipeline._prepare_latent_images(x_0)

        # Get text conditioning
        t5_tokens, clip_tokens = self.pipeline.tokenize(request.prompt)
        txt, txt_ids, vec = self.pipeline._prepare_conditioning(1, t5_tokens, clip_tokens)

        # strength=0 means no noise (keep original), strength=1 means full noise
        t_start = 1.0 - request.strength

        # Add noise to the encoded latents
        noise = self.pipeline.sampler.sample_prior(x_0.shape, dtype=x_0.dtype)
        x_t = self.pipeline.sampler.add_noise(x_0, t_start, noise=noise)

        # Run denoising loop from t_start to 0
        for x_t in self.pipeline._denoising_loop(
            x_t, x_ids, txt, txt_ids, vec,
            num_steps=request.num_inference_steps,
            guidance=request.guidance_scale,
            start=t_start,
            stop=0,
        ):
            mx.eval(x_t)

        # Decode latents to image
        decoded = self.pipeline.decode(x_t, latent_size=latent_size)
        mx.eval(decoded)

        img_array = np.array(decoded[0])
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

        return Image2ImageResponse(
            output_path=output_path,
            metadata={
                "provider": "mlx",
                "model": self.model_name,
                "mode": "img2img",
                "seed": request.seed,
                "strength": request.strength,
                "width": request.width,
                "height": request.height,
            },
        )
```

- [ ] **Step 2: Verify the provider file has no syntax errors**

Run: `uv run python -c "import ast; ast.parse(open('packages/image2image/src/image2image/providers/mlx.py').read()); print('Syntax OK')"`
Expected: Syntax OK

- [ ] **Step 3: Commit the provider**

```bash
git add packages/image2image/src/image2image/providers/mlx.py
git commit -m "feat: add MLX FLUX provider with text-to-image and image-to-image support"
```

---

### Task 4: Register Provider and Update Dependencies

**Files:**
- Modify: `packages/image2image/src/image2image/providers/__init__.py`
- Modify: `packages/image2image/pyproject.toml`

- [ ] **Step 1: Update provider registration**

Modify `packages/image2image/src/image2image/providers/__init__.py`:

```python
from aiservices_core.providers import registry

from .comfyui import ComfyUIProvider
from .replicate_cloud import ReplicateProvider

registry.register("image2image.comfyui", ComfyUIProvider)
registry.register("image2image.replicate", ReplicateProvider)

try:
    from .mlx import MLXFluxProvider

    registry.register("image2image.mlx", MLXFluxProvider)
except ImportError:
    # MLX dependencies not available — skip registration
    pass
```

The `try/except` ensures the package still works on machines without MLX installed (e.g., Linux CI).

- [ ] **Step 2: Update pyproject.toml dependencies**

Modify `packages/image2image/pyproject.toml` to add MLX dependencies:

```toml
[project]
name = "image2image"
version = "0.1.0"
description = "Image to image generation pipeline"
requires-python = ">=3.10"
dependencies = [
    "aiservices-core",
    "replicate>=0.25.0",
    "requests>=2.31.0",
    "pillow>=10.0.0",
    "pydantic>=2.0.0",
    "typer>=0.12.0",
    "mlx>=0.21.0",
    "huggingface-hub>=0.20.0",
    "sentencepiece>=0.2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project.scripts]
image2image = "image2image.cli:app"

[tool.uv.sources]
aiservices-core = { workspace = true }
```

- [ ] **Step 3: Sync dependencies**

Run: `uv sync`
Expected: Dependencies resolve successfully

- [ ] **Step 4: Verify registration works**

Run: `uv run python -c "from image2image.providers import registry; print(registry._providers.keys())"`
Expected: Should include `image2image.mlx` (on Apple Silicon) or just `image2image.comfyui` and `image2image.replicate` (on other platforms)

- [ ] **Step 5: Commit registration and deps**

```bash
git add packages/image2image/src/image2image/providers/__init__.py packages/image2image/pyproject.toml
git commit -m "feat: register MLX provider, add mlx dependencies"
```

---

### Task 5: Update CLI

**Files:**
- Modify: `packages/image2image/src/image2image/cli.py`

- [ ] **Step 1: Update CLI to support optional input and dimensions**

Modify `packages/image2image/src/image2image/cli.py`:

```python
import time

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger

from .models import Image2ImageRequest
from .providers import registry

app = typer.Typer(help="Image to image generation pipeline")
logger = get_logger(__name__)


@app.command()
def main(
    input_path: str = typer.Option(None, "--input", "-i", help="Path to input image (omit for text-to-image)"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Text prompt"),
    output_path: str = typer.Option(..., "--output", "-o", help="Path to save output"),
    provider_name: str = typer.Option(
        "image2image.comfyui",
        "--provider",
        help="Provider name",
    ),
    strength: float = typer.Option(0.5, "--strength", "-s", help="Transformation strength (image-to-image only)"),
    guidance_scale: float = typer.Option(7.5, "--guidance", help="Guidance scale"),
    steps: int = typer.Option(50, "--steps", help="Number of inference steps"),
    seed: int = typer.Option(None, "--seed", help="Random seed"),
    negative_prompt: str = typer.Option(
        None, "--negative-prompt", "-n", help="Negative text prompt"
    ),
    width: int = typer.Option(1024, "--width", help="Output width in pixels"),
    height: int = typer.Option(1024, "--height", help="Output height in pixels"),
    verbose: bool = verbose_option,
    device: str = device_option,
):
    try:
        request = Image2ImageRequest(
            image_path=input_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            seed=seed,
            width=width,
            height=height,
        )

        logger.info(f"Using provider: {provider_name}")

        with create_progress_bar() as progress:
            task_id = progress.add_task("[cyan]Initializing provider...", total=None)

            provider = registry.get(provider_name, device=device)

            progress.update(task_id, description="[green]Generating image...")
            start_time = time.time()

            response = provider.generate(request, output_path)

            elapsed = time.time() - start_time
            progress.update(task_id, description=f"[bold green]Done in {elapsed:.2f}s!")

        logger.info(f"Image saved to: {response.output_path}")
        logger.debug(f"Metadata: {response.metadata}")

    except Exception as e:
        logger.exception(f"Generation failed: {str(e)}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Verify CLI help works**

Run: `uv run image2image --help`
Expected: Shows help with all options including `--width`, `--height`, and optional `--input`

- [ ] **Step 3: Commit CLI changes**

```bash
git add packages/image2image/src/image2image/cli.py
git commit -m "feat: make CLI input optional, add width/height options"
```

---

### Task 6: Extend Model Loading for Custom HuggingFace Model IDs

**Files:**
- Modify: `packages/image2image/src/image2image/flux_mlx/utils.py`

- [ ] **Step 1: Add support for custom model IDs in utils.py**

The current `utils.py` has a hardcoded `configs` dict with only "flux-schnell" and "flux-dev". Add a function to resolve arbitrary HuggingFace model IDs:

Add to the end of `packages/image2image/src/image2image/flux_mlx/utils.py`:

```python
def resolve_model_name(name: str) -> str:
    """Resolve a model name to a known config key.

    Supports:
    - Built-in names: "flux-schnell", "flux-dev"
    - HuggingFace IDs: "mlx-community/FLUX.1-schnell" -> "flux-schnell"
    - Auto-detection: any ID containing "schnell" -> "flux-schnell", "dev" -> "flux-dev"
    """
    if name in configs:
        return name

    name_lower = name.lower()
    if "schnell" in name_lower:
        return "flux-schnell"
    if "dev" in name_lower:
        return "flux-dev"

    raise ValueError(
        f"Unknown model: {name}. Supported: {list(configs.keys())}. "
        f"For custom models, use a name containing 'schnell' or 'dev'."
    )
```

- [ ] **Step 2: Update FluxPipeline to use resolve_model_name**

In `packages/image2image/src/image2image/flux_mlx/flux.py`, update the `__init__` method to resolve the model name:

```python
def __init__(self, name: str, t5_padding: bool = True):
    from .utils import resolve_model_name

    self.dtype = mx.bfloat16
    self.name = resolve_model_name(name)
    self.t5_padding = t5_padding

    self.ae = load_ae(self.name)
    self.flow = load_flow_model(self.name)
    self.clip = load_clip(self.name)
    self.clip_tokenizer = load_clip_tokenizer(self.name)
    self.t5 = load_t5(self.name)
    self.t5_tokenizer = load_t5_tokenizer(self.name)
    self.sampler = FluxSampler(self.name)
```

- [ ] **Step 3: Commit model loading changes**

```bash
git add packages/image2image/src/image2image/flux_mlx/utils.py packages/image2image/src/image2image/flux_mlx/flux.py
git commit -m "feat: support custom HuggingFace model IDs in FLUX pipeline"
```

---

### Task 7: Tests

**Files:**
- Create: `packages/image2image/tests/test_mlx_provider.py`

- [ ] **Step 1: Write unit test for _to_latent_size**

Create `packages/image2image/tests/test_mlx_provider.py`:

```python
import pytest


def test_to_latent_size():
    from image2image.providers.mlx import _to_latent_size

    # Exact multiples of 16
    assert _to_latent_size((256, 256)) == (32, 32)
    assert _to_latent_size((512, 512)) == (64, 64)

    # Non-multiples get rounded up to next multiple of 16, then divided by 8
    # 300 -> 304 -> 38, 400 -> 400 -> 50
    assert _to_latent_size((300, 400)) == (38, 50)
```

- [ ] **Step 2: Write test for MLXFluxProvider instantiation**

```python
@pytest.mark.skipif(
    not _mlx_available(),
    reason="MLX not available",
)
def test_mlx_provider_init():
    from image2image.providers.mlx import MLXFluxProvider

    provider = MLXFluxProvider(model_name="flux-schnell")
    assert provider.model_name == "flux-schnell"
    assert provider._pipeline is None  # Lazy loaded
```

Add a helper at the top of the file:

```python
def _mlx_available():
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest packages/image2image/tests/ -v`
Expected: All model tests pass. MLX provider tests are skipped if mlx is not installed.

- [ ] **Step 4: Commit tests**

```bash
git add packages/image2image/tests/test_mlx_provider.py packages/image2image/tests/test_models.py
git commit -m "test: add tests for MLX provider and optional image_path model"
```

---

### Task 8: Integration Verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest packages/image2image/ -v`
Expected: All tests pass

- [ ] **Step 2: Run linter**

Run: `uvx ruff check packages/image2image/`
Expected: No errors

- [ ] **Step 3: Run type checker**

Run: `uvx pyright packages/image2image/`
Expected: No type errors (or only pre-existing ones)

- [ ] **Step 4: Test CLI end-to-end (on Apple Silicon only)**

Run: `uv run image2image --provider image2image.mlx --prompt "a red sphere on a white background" --output /tmp/test_mlx.png --width 256 --height 256 --steps 2`
Expected: Image saved to /tmp/test_mlx.png (first run will download the model)

- [ ] **Step 5: Final commit with all changes**

```bash
git add -A
git status  # Verify all changes are staged
git commit -m "feat: complete MLX FLUX provider for image2image package"
```
