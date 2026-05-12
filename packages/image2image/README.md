# image2image

Image-to-Image generation module for AIServices. Uses FLUX.2-klein-9B with MLX as the local backend on Apple Silicon.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/image2image
```

## CLI Usage

```bash
image2image --input photo.jpg --prompt "turn into watercolor painting" --output out.png
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input`, `-i` | (required) | Path to input image |
| `--prompt`, `-p` | (required) | Text prompt for generation |
| `--output`, `-o` | (required) | Path to save output image |
| `--provider` | `image2image.mlx` | Provider to use |
| `--strength`, `-s` | 0.5 | Transformation strength (0.0-1.0) |
| `--guidance` | 7.5 | Guidance scale (CFG) |
| `--steps` | 50 | Number of inference steps |
| `--seed` | random | Random seed |
| `--negative-prompt`, `-n` | None | Negative text prompt |

## Python API

```python
from image2image.client import generate

output_path = generate(
    image_path="photo.jpg",
    prompt="turn into watercolor painting",
    output_path="output.png",
)
```
