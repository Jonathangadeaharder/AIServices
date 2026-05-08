# video2subtitle

Video-to-Subtitle transcription module for AIServices. Extracts audio via FFmpeg, then transcribes with mlx-whisper (Apple Silicon) to SRT/VTT.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/video2subtitle
```

## CLI Usage

```bash
video2subtitle --video movie.mp4 --output subtitles.srt
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--video` | (required) | Path to input video file |
| `--output`, `-o` | (required) | Path to save subtitle file |
| `--format`, `-f` | `srt` | Output format (srt or vtt) |
| `--language`, `-l` | auto | Language code (e.g. 'en', 'de') |
| `--model` | `mlx-community/whisper-large-v3-turbo` | Whisper model name |
| `--no-word-timestamps` | off | Disable word-level timestamps |
| `--provider` | `video2subtitle.mlx` | Provider to use |

## Python API

```python
from video2subtitle.client import transcribe

output_path = transcribe(
    video_path="movie.mp4",
    output_path="subtitles.srt",
    language="en",
)
```
