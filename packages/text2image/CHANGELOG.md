# Changelog — text2image

All notable changes to this module will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-06

### Added

- Text-to-image generation with Flux 2 MLX on Apple Silicon
- `text2image.mlx` provider for local MLX inference
- CLI: `text2image generate --prompt <text> --output <img>`
- Python API: `text2image.client.generate()`
- Configurable dimensions (512–2048, divisible by 8), guidance scale, inference steps, seed
- Pydantic request/response models with dimension validation
- E2E tests
