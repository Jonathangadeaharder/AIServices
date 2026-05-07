# Changelog — text2video

All notable changes to this module will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-06

### Added

- Text-to-video generation with LTX 2.3 MLX on Apple Silicon
- `text2video.mlx` provider for local MLX inference
- CLI: `text2video generate --prompt <text> --output <mp4>`
- Python API via provider pattern
- Configurable dimensions, frame count, inference steps, FPS, seed
- Pydantic request/response models
- E2E tests
