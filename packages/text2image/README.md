# text2image

Text-to-Image generation module for AIServices. Uses MLX for local inference on Apple Silicon.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/text2image
```

## Usage

You can run the module via the CLI:

```bash
text2image generate --prompt "A futuristic cityscape at sunset" --output out.png --width 1024 --height 1024
```

Available providers:
- `text2image.mlx`: Local MLX inference using FLUX.2-klein-9B (default)
