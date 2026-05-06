# video2audio

Extract audio from video files using FFmpeg.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/video2audio
```

## Usage

```bash
video2audio --input video.mp4 --output audio.wav
video2audio --input video.mp4 --output audio.mp3 --codec mp3
video2audio --help
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | *(required)* | Path to input video file |
| `--output`, `-o` | *(required)* | Path to save output audio |
| `--codec`, `-c` | `wav` | Output audio format (wav, mp3, aac) |
| `--provider` | `video2audio.ffmpeg` | Provider name |
| `--verbose`, `-v` | `false` | Enable verbose output |

## Providers

- `video2audio.ffmpeg`: Audio extraction using FFmpeg.

## Requirements

- FFmpeg installed and available on PATH
