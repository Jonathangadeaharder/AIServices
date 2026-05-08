# image2video

Image-to-Video generation module for AIServices. Uses LTX-Video 2.3 with MLX as the local backend on Apple Silicon.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/image2video
```

## CLI Usage

```bash
image2video --image photo.jpg --prompt "zoom into the landscape" --output out.mp4
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--image`, `-i` | (required) | Path to input image |
| `--prompt`, `-p` | (required) | Text prompt for video generation |
| `--output`, `-o` | (required) | Path to save output video |
| `--provider` | `image2video.mlx` | Provider to use |
| `--negative-prompt`, `-n` | (default) | Negative prompt |
| `--width` | 640 | Video width (must be divisible by 8) |
| `--height` | 640 | Video height (must be divisible by 8) |
| `--frames` | 81 | Number of frames to generate |
| `--steps` | 4 | Number of inference steps |
| `--seed` | random | Random seed |
| `--fps` | 16 | Frames per second |

## Python API

```python
from image2video.client import generate

output_path = generate(
    image_path="photo.jpg",
    prompt="zoom into the landscape",
    output_path="output.mp4",
    num_frames=81,
    fps=16,
)
```
