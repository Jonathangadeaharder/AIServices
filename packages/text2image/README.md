# text2image

Text-to-Image generation module for AIServices. Uses ComfyUI as the local backend.

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
- `text2image.comfyui`: Uses a running ComfyUI server with the Flux2 workflow (default)
- `text2image.replicate`: Uses Replicate Cloud (Requires `REPLICATE_API_TOKEN` environment variable)

## Prerequisites

For the `comfyui` provider, you need a running ComfyUI server at `127.0.0.1:8188` with the Flux2 models installed.
