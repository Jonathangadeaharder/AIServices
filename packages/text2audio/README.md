# text2audio

Text-to-Audio generation module for AIServices. Generates music, SFX, and ambient audio from text prompts.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/text2audio
```

## CLI Usage

```bash
text2audio --prompt "calm piano music" --output out.wav
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt`, `-p` | (required) | Text prompt describing the audio |
| `--output`, `-o` | (required) | Path to save output audio file |
| `--category`, `-c` | `music` | Audio category (music, sfx, ambient, speech) |
| `--duration` | 10.0 | Duration in seconds |
| `--format`, `-f` | `wav` | Output format (wav, mp3) |
| `--seed`, `-s` | random | Random seed |
| `--provider` | `text2audio.mlx` | Provider to use |

## Python API

```python
from text2audio.client import generate

output_path = generate(
    prompt="calm piano music",
    output_path="output.wav",
    category="music",
    duration_seconds=10.0,
)
```
