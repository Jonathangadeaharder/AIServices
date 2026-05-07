# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-06

### Added

- Initial monorepo structure with uv workspace
- `aiservices-core` shared library (config, provider registry, logging, IO)
- `image2image` — Flux 2 MLX image-to-image editing
- `text2image` — Flux 2 MLX text-to-image generation
- `image2video` — LTX 2.3 MLX image-to-video generation
- `text2video` — LTX 2.3 MLX text-to-video generation
- `video2audio` — FFmpeg audio extraction
- `audio2subtitle` — Whisper MLX audio transcription to SRT/VTT
- `text2audio` — Fish Speech S2 Pro MLX audio generation (placeholder)
- `text2speech` — Fish Speech text-to-speech synthesis
- `video2subtitle` — Pipeline: video → audio → subtitle
- `subtitle-translate` — SRT/VTT translation
- `subtitle-filter` — SRT/VTT filtering
- CLI entry points for all service modules
- End-to-end test suite with pytest
- Provider pattern for pluggable ML backends (MLX, ComfyUI, Replicate)
- Python API via client modules (`<module>.client.generate()`)
- Examples directory with runnable scripts per module
