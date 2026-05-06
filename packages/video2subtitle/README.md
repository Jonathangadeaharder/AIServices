# video2subtitle

Video to subtitle transcription pipeline. Extracts audio from video and transcribes it using mlx-whisper.

## Install

```bash
uv tool install ./packages/video2subtitle
```

Or install all packages:

```bash
uv sync --all-packages
```

Requires `ffmpeg` on PATH.

## Usage

```bash
video2subtitle --input video.mp4 --output subtitles.srt
video2subtitle --input video.mp4 --language en
video2subtitle --input video.mp4 --burn-in
video2subtitle --help
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | required | Input video path |
| `--output`, `-o` | `<input>.srt` | Output subtitle path |
| `--language`, `-l` | auto-detect | Language code (e.g. `en`) or `auto` |
| `--burn-in` | off | Burn subtitles into a new mp4 file |
| `--verbose` | off | Verbose logging |
| `--device` | `auto` | Compute device |

## Architecture

Pipeline: video → ffmpeg (audio extraction) → mlx-whisper (transcription) → SRT file.

Optional: `--burn-in` uses ffmpeg to hard-code subtitles into the video.

Provider pattern via `aiservices-core` registry. Default provider: `video2subtitle.mlx`.
