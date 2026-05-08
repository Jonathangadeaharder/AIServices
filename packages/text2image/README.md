# text2image

Text-to-Image generation module for AIServices. Uses FLUX-schnell with MLX as the local backend on Apple Silicon.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/text2image
```

## CLI Usage

```bash
text2image --prompt "a beautiful sunset over mountains" --output out.png
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt`, `-p` | (required) | Text prompt for image generation |
| `--output`, `-o` | (required) | Path to save output image |
| `--provider` | `text2image.mlx` | Provider to use |
| `--negative-prompt` | None | Negative prompt |
| `--guidance-scale` | 7.5 | Guidance scale (CFG) |
| `--steps` | 50 | Number of inference steps |
| `--seed` | random | Random seed |
| `--width` | 1024 | Image width (must be divisible by 8, >= 512) |
| `--height` | 1024 | Image height (must be divisible by 8, >= 512) |

## Python API

```python
from text2image.client import generate

output_path = generate(
    prompt="a beautiful sunset over mountains",
    output_path="output.png",
    width=1024,
    height=1024,
)
```
