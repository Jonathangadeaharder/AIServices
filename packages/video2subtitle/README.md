# video2subtitle

Video to subtitle (SRT/VTT) transcription pipeline. Extracts audio from video and transcribes it using mlx-whisper.

## Install

```bash
uv sync --all-packages
```

Requires `ffmpeg` on PATH.

## Usage

```bash
video2subtitle transcribe --video input.mp4 --output subtitles.srt
video2subtitle transcribe --video input.mp4 --output subtitles.vtt --format vtt
video2subtitle transcribe --video input.mp4 --output out.srt --language en --model mlx-community/whisper-large-v3-turbo
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | required | Input video path |
| `--output` | required | Output subtitle path |
| `--format` | `srt` | Output format (`srt` or `vtt`) |
| `--language` | auto-detect | Language code |
| `--model` | `mlx-community/whisper-large-v3-turbo` | Whisper model |
| `--no-word-timestamps` | off | Disable word-level timestamps |
| `--provider` | `video2subtitle.mlx` | Provider name |
| `--verbose` | off | Verbose logging |
| `--device` | `auto` | Compute device |

## Architecture

Pipeline: video → ffmpeg (audio extraction) → mlx-whisper (transcription) → SRT/VTT file.

Provider pattern via `aiservices-core` registry. Default provider: `video2subtitle.mlx`.
