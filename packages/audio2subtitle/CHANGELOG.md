# Changelog — audio2subtitle

All notable changes to this module will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-06

### Added

- Audio transcription to SRT/VTT using Whisper MLX on Apple Silicon
- `audio2subtitle.mlx` provider for local MLX inference
- CLI: `audio2subtitle transcribe --audio <wav> --output <srt>`
- Python API: `audio2subtitle.client.generate()`
- Word-level timestamps with mlx-whisper
- Configurable model, language (auto-detect), output format
- Pydantic request/response models with SubtitleEntry
- E2E tests
