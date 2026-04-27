# Image2Image Module

Image-to-image generation pipeline module for AIServices.

## Installation

You can install this module using `uv`:

```bash
uv tool install ./packages/image2image
```

## Usage

```bash
image2image --input in.jpg --prompt "watercolor painting" --output out.jpg --provider local_sdxl
```

### Providers
- `local_sdxl`: Uses `diffusers` locally (optimized for MPS/CUDA/CPU).
- `replicate`: Uses Replicate API (requires `REPLICATE_API_TOKEN`).
