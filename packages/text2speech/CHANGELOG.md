# Changelog — text2speech

All notable changes to this module will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-06

### Added

- Text-to-speech synthesis using Fish Speech on Apple Silicon
- `text2speech.fish_mlx` provider for local MLX inference
- CLI: `text2speech generate --text <text> --output <wav>`
- Python API: `text2speech.client.generate()`
- Voice cloning via reference audio
- Emotion, tone, and effect tags
- Pydantic request/response models
- E2E tests
