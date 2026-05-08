# video2audio

Video-to-Audio extraction module for AIServices. Uses FFmpeg to extract audio tracks from video files.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/video2audio
```

## CLI Usage

```bash
video2audio --video input.mp4 --output audio.wav
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--video`, `-i` | (required) | Path to input video file |
| `--output`, `-o` | (required) | Path to save output audio |
| `--format`, `-f` | `wav` | Output audio format (wav, mp3, aac) |
| `--sample-rate`, `-r` | 44100 | Output sample rate in Hz |
| `--mono/--stereo` | mono | Convert to mono or keep stereo |
| `--provider` | `video2audio.ffmpeg` | Provider to use |

## Python API

```python
from video2audio.client import extract

output_path = extract(
    video_path="input.mp4",
    output_path="audio.wav",
    output_format="wav",
    sample_rate=44100,
)
```
