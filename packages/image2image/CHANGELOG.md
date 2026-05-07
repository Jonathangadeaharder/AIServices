# Changelog — image2image

All notable changes to this module will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-06

### Added

- Image-to-image generation with Flux 2 MLX on Apple Silicon
- `image2image.mlx` provider for local MLX inference
- CLI: `image2image --input <img> --prompt <text> --output <img>`
- Python API: `image2image.client.generate()`
- Configurable strength, guidance scale, inference steps, seed
- Pydantic request/response models
- E2E tests
