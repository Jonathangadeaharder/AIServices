# speech2text

Speech-to-Text transcription module for AIServices. Uses [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for fast local inference on Apple Silicon.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/speech2text
```

## Usage

```bash
speech2text transcribe --audio recording.wav --output transcript.txt
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--audio`, `-a` | *(required)* | Path to input audio file |
| `--output`, `-o` | `None` | Path to save transcription text |
| `--model` | `mlx-community/whisper-large-v3-mlx` | Whisper model name |
| `--language`, `-l` | `None` | Language code (e.g. `en`) |
| `--provider` | `speech2text.mlx` | Provider name |
| `--verbose`, `-v` | `false` | Enable verbose output |

## Providers

- `speech2text.mlx`: Local transcription using mlx-whisper (Apple Silicon optimised)

## Additional modules

### `mlx_marian` — MLX-native MarianMT translation

A standalone MLX implementation of the MarianMT sequence-to-sequence model for translation. Includes:

- Encoder-decoder transformer architecture
- Hugging Face → MLX weight conversion
- Greedy decoding
