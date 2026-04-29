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
| `image2image` | Image-to-image editing (MLX Flux, ComfyUI, Replicate) | `image2image` |
| `text2image` | Text-to-image generation (ComfyUI Flux2, Replicate) | `text2image` |
| `text2video` | Text-to-video generation (MLX LTX-2.3, ComfyUI Wan 2.2) | `text2video` |
| `speech2text` | Speech-to-text transcription (MLX Whisper) | `speech2text` |
| `text2speech` | Text-to-speech synthesis (Fish Speech) | `text2speech` |
| `studio` | Episode orchestrator for sitcom pilot generation | `studio` |
| `ltx-core-mlx` | Apple Silicon MLX video generation engine (vendored) | -- |
| `ltx-pipelines-mlx` | LTX-2.3 generation pipelines for MLX (vendored) | -- |
| `fish-speech` | Fish Speech TTS core (vendored) | -- |

## Development

```bash
uvx ruff check          # lint
uvx ruff format         # format
uvx pyright             # type check
uv run pytest           # run tests
```

All packages follow the same layout: `src/<package_name>/` with a `cli.py` entry point registered in `pyproject.toml`.
