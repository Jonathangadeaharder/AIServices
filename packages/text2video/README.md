# text2video

Text-to-Video generation module for AIServices. Uses LTX-Video 2.3 with MLX as the local backend on Apple Silicon.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/text2video
```

## CLI Usage

```bash
text2video --prompt "A red ball bouncing on a wooden floor" --output out.mp4
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt`, `-p` | (required) | Text prompt for video generation |
| `--output`, `-o` | (required) | Path to save output video |
| `--provider` | `text2video.mlx` | Provider to use |
| `--negative-prompt`, `-n` | (default) | Negative prompt |
| `--seconds` | 4 | Video duration in seconds |
| `--fps` | 24 | Frames per second |
| `--width` | 704 | Video width (minimum 64, must be divisible by 8) |
| `--height` | 480 | Video height (minimum 64, must be divisible by 8) |
| `--steps` | 8 | Number of inference steps |
| `--seed` | random | Random seed |

## Python API

```python
from text2video.client import generate

output_path = generate(
    prompt="A red ball bouncing on a wooden floor",
    output_path="output.mp4",
    seconds=4,
    fps=24,
)
```
