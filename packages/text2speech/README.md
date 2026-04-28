# text2speech

Text-to-Speech generation module for AIServices. Uses [Fish Speech](https://github.com/fishaudio/fish-speech) for local high-quality voice synthesis.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/text2speech
```

## Usage

```bash
text2speech generate --text "Hello, how can I help you today?" --output speech.wav
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--text`, `-t` | *(required)* | Text to convert to speech |
| `--output`, `-o` | *(required)* | Path to save output audio file |
| `--provider` | `text2speech.fish` | Provider name |
| `--verbose`, `-v` | `false` | Enable verbose output |

## Providers

- `text2speech.fish`: Local synthesis using Fish Speech models.
