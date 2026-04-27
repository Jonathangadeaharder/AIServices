# text2image

Text-to-Image generation module for AIServices.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/text2image
```

## Usage

You can run the module via the CLI:

```bash
text2image --prompt "A futuristic cityscape at sunset" --output out.png --width 1024 --height 1024 --provider local_sdxl
```

Available providers:
- `local_sdxl`: Uses Diffusers SDXL Base locally (Requires 16GB+ RAM)
- `replicate`: Uses Replicate Cloud (Requires `REPLICATE_API_TOKEN` environment variable)
