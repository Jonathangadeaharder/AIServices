# audio2subtitle

Audio-to-subtitle (SRT/VTT) transcription pipeline for AIServices. Uses [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for fast local inference on Apple Silicon with word-level timestamps.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/audio2subtitle
```

## Usage

```bash
audio2subtitle --input recording.wav --output subtitles.srt
audio2subtitle -i recording.wav -o subtitles.srt --language en
audio2subtitle -i recording.wav -o subtitles.srt --model mlx-community/whisper-large-v3
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | *(required)* | Path to input audio file |
| `--output`, `-o` | *(required)* | Path to output subtitle file |
| `--language`, `-l` | `None` | Language code (e.g. `en`), auto-detect if omitted |
| `--model` | `mlx-community/whisper-large-v3` | Whisper model name |

## Output Format

SRT with word-level timestamps:

```
1
00:00:00,000 --> 00:00:02,500
Hello world.

2
00:00:02,500 --> 00:00:05,000
Goodbye.
```

## Provider

Uses `audio2subtitle.mlx` — local transcription via mlx-whisper (Apple Silicon optimised).
