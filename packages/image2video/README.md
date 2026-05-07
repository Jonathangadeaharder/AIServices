# image2video

Image-to-Video generation module for AIServices. Uses LTX-Video with MLX as the local backend on Apple Silicon.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/image2video
```

## CLI Usage

```bash
image2video --input input.png --output out.mp4 [--seconds 4] [--fps 24] [--prompt "text"]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input`, `-i` | (required) | Path to the input image |
| `--output`, `-o` | (required) | Path to save output video |
| `--seconds`, `-s` | 4 | Video duration in seconds |
| `--fps` | 24 | Frames per second |
| `--prompt`, `-p` | "" | Text prompt for video generation |
| `--provider` | `image2video.mlx` | Provider to use |
| `--negative-prompt`, `-n` | (default) | Negative prompt |
| `--width` | 640 | Video width (must be divisible by 8) |
| `--height` | 640 | Video height (must be divisible by 8) |
| `--steps` | 4 | Number of inference steps |
| `--seed` | random | Random seed |

## Python API

```python
from image2video.models import Image2VideoRequest
from image2video.providers import registry

request = Image2VideoRequest(
    image_path="input.png",
    prompt="A cinematic drone shot over a mountain lake",
)

provider = registry.get("image2video.mlx")
response = provider.generate(request, "output.mp4")
print(response.output_path)
```
