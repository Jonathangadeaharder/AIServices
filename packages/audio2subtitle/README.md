# audio2subtitle

Audio-to-subtitle (SRT/VTT) transcription pipeline for AIServices. Uses [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for fast local inference on Apple Silicon with word-level timestamps.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/audio2subtitle
```

## Usage

```bash
audio2subtitle transcribe --audio recording.wav --output subtitles.srt
audio2subtitle transcribe --audio recording.wav --output captions.vtt --format vtt
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--audio`, `-a` | *(required)* | Path to input audio file |
| `--output`, `-o` | *(required)* | Path to output subtitle file |
| `--format`, `-f` | `srt` | Output format: `srt` or `vtt` |
| `--language`, `-l` | `None` | Language code (e.g. `en`), auto-detect if omitted |
| `--model` | `mlx-community/whisper-large-v3-turbo` | Whisper model name |
| `--no-word-timestamps` | `false` | Disable word-level timestamps |
| `--provider` | `audio2subtitle.mlx` | Provider name |
| `--verbose`, `-v` | `false` | Enable verbose output |

## Output Formats

### SRT
```
1
00:00:00,000 --> 00:00:02,500
Hello world.

2
00:00:02,500 --> 00:00:05,000
Goodbye.
```

### VTT
```
WEBVTT

00:00:00.000 --> 00:00:02.500
Hello world.

00:00:02.500 --> 00:00:05.000
Goodbye.
```

## Providers

- `audio2subtitle.mlx`: Local transcription using mlx-whisper (Apple Silicon optimised)
