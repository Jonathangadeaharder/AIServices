# Image2Image Module

Image-to-image generation pipeline module for AIServices.

## Installation

You can install this module using `uv`:

```bash
uv tool install ./packages/image2image
```

## Usage

```bash
image2image --input in.jpg --prompt "watercolor painting" --output out.jpg
```

### Providers
- `image2image.mlx`: Local MLX inference using FLUX.2-klein-9B on Apple Silicon.
