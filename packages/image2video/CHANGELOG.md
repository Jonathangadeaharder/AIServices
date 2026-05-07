# Changelog — image2video

All notable changes to this module will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-06

### Added

- Image-to-video generation with LTX 2.3 MLX on Apple Silicon
- `image2video.mlx` provider for local MLX inference
- CLI: `image2video generate --image <img> --prompt <text> --output <mp4>`
- Python API: `image2video.client.generate()`
- Configurable dimensions, frame count, inference steps, FPS, seed
- Pydantic request/response models with dimension validation
- E2E tests
