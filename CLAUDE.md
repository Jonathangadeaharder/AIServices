# AIServices Monorepo

uv-workspace Python monorepo. Ten packages under `packages/`. Shared core, individual CLIs, vendored ML engines.

## Commands

```bash
uv sync --all-packages              # install everything
uv run pytest                       # run all tests
uvx ruff check                      # lint
uvx ruff format                     # format
uvx pyright                         # type check
```

## Architecture

- Root `pyproject.toml` declares uv workspace with all member packages.
- `packages/aiservices-core` holds shared config, provider registry, logging, IO.
- Each service package (`image2image`, `text2image`, `text2video`, `speech2text`, `text2speech`, `studio`) has its own `pyproject.toml`, `src/` layout, and CLI entry point.
- `packages/ltx-core-mlx`, `packages/ltx-pipelines-mlx`, `packages/fish_speech` are vendored upstream engines with local adaptations.
- Provider pattern: each service exposes providers (MLX local, ComfyUI API, Replicate API) selected via config.
