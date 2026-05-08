# AI Services

Monorepo for AI-powered media generation services. Image, video, speech, and orchestration tools built on Apple Silicon MLX and cloud providers.

## Architecture

- **uv workspace** -- single `pyproject.toml` at root defines all packages
- **`aiservices-core`** -- shared config, provider registry, logging, IO utilities
- **Individual packages** -- each in `packages/`, own `pyproject.toml`, own CLI entry point
- **Vendored engines** -- `ltx-core-mlx`, `ltx-pipelines-mlx`, `fish_speech` ship upstream source with local adaptations

## Quick Start

```bash
uv sync --all-packages
uv run pytest
```

## Packages

| Package | Description | CLI |
|---|---|---|
| `aiservices-core` | Shared config, providers, logging, IO | -- |
| `image2image` | Image-to-image editing (MLX Flux) | `image2image` |
| `text2image` | Text-to-image generation (MLX Flux) | `text2image` |
| `image2video` | Image-to-video generation (MLX LTX-2.3) | `image2video` |
| `text2video` | Text-to-video generation (MLX LTX-2.3) | `text2video` |
| `video2audio` | Audio extraction from video (FFmpeg) | `video2audio` |
| `audio2subtitle` | Audio-to-SRT/VTT transcription (MLX Whisper) | `audio2subtitle` |
| `video2subtitle` | Video-to-SRT/VTT transcription (FFmpeg + MLX Whisper) | `video2subtitle` |
| `text2audio` | Text-to-audio generation (music, SFX, ambient) | `text2audio` |
| `text2speech` | Text-to-speech synthesis (Fish Speech) | `text2speech` |
| `ltx-core-mlx` | Apple Silicon MLX video generation engine (vendored) | -- |
| `ltx-pipelines-mlx` | LTX-2.3 generation pipelines for MLX (vendored) | -- |

## Development

```bash
uvx ruff check          # lint
uvx ruff format         # format
uvx pyright             # type check
uv run pytest           # run tests
```

All packages follow the same layout: `src/<package_name>/` with a `cli.py` entry point registered in `pyproject.toml`.
