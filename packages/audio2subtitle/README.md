# audio2subtitle

Audio-to-Subtitle transcription module for AIServices. Uses mlx-whisper (Apple Silicon) for local speech-to-text with SRT/VTT output.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/audio2subtitle
```

## CLI Usage

```bash
audio2subtitle --audio speech.wav --output subtitles.srt
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--audio`, `-a` | (required) | Path to input audio file |
| `--output`, `-o` | (required) | Path to output subtitle file |
| `--format`, `-f` | `srt` | Output format (srt or vtt) |
| `--language`, `-l` | auto | Language code (e.g. 'en', 'de') |
| `--model` | `mlx-community/whisper-large-v3-turbo` | Whisper model name |
| `--no-word-timestamps` | off | Disable word-level timestamps |
| `--provider` | `audio2subtitle.mlx` | Provider to use |

## Python API

```python
from audio2subtitle.client import generate

output_path = generate(
    audio_path="speech.wav",
    output_path="subtitles.srt",
    language="en",
)
```
