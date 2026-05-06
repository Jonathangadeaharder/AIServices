# text2image

Text-to-Image generation module for AIServices. Uses MLX for local inference on Apple Silicon.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/text2image
```

## Usage

Generate an image from a text prompt:

```bash
text2image --prompt "A futuristic cityscape at sunset" --output out.png --width 1024 --height 1024
```

Options:
- `--prompt`, `-p`: Text prompt for image generation (required)
- `--output`, `-o`: Path to save output image (required)
- `--width`: Width of the image (default: 1024)
- `--height`: Height of the image (default: 1024)
- `--seed`: Random seed for reproducibility
- `--steps`: Number of inference steps (default: 50)
- `--guidance-scale`: Guidance scale (default: 7.5)
- `--negative-prompt`: Negative text prompt
- `--provider`: Provider name (default: text2image.mlx)

Available providers:
- `text2image.mlx`: Local MLX inference using FLUX.2-klein-9B (default)
