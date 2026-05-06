# Changelog — video2audio

All notable changes to this module will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-06

### Added

- Audio extraction from video files using FFmpeg
- `video2audio.ffmpeg` provider
- CLI: `video2audio extract --video <mp4> --output <wav>`
- Python API: `video2audio.models.Video2AudioRequest` + provider
- Configurable output format (wav, mp3, aac), sample rate, mono/stereo
- Pydantic request/response models
- E2E tests
