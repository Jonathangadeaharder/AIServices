# Changelog — video2subtitle

All notable changes to this module will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-06

### Added

- Video-to-subtitle pipeline: video → FFmpeg audio extraction → Whisper MLX transcription
- `video2subtitle.mlx` provider
- CLI: `video2subtitle transcribe --video <mp4> --output <srt>`
- Python API: `video2subtitle.models.Video2SubtitleRequest` + provider
- Word-level timestamps
- Configurable model, language (auto-detect), output format (srt, vtt)
- Pydantic request/response models with SubtitleEntry
- E2E tests
