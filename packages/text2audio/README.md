# text2audio

Text-to-audio generation pipeline for the AIServices monorepo.

## Installation

```bash
uv tool install ./packages/text2audio
```

## Usage

### CLI

```bash
text2audio --text "calm piano music" --output out.wav
text2audio --text "thunder and rain" --output rain.wav --voice speaker1 --speed 1.2
text2audio --help
```

### Python

```python
from text2audio.models import Text2AudioRequest
from text2audio.providers import registry

request = Text2AudioRequest(
    text="calm piano music",
    voice="default",
    speed=1.0,
)

provider = registry.get("text2audio.fish_mlx")
response = provider.generate(request, "output.wav")
```

## Providers

- `text2audio.fish_mlx` - Fish S2 Pro MLX (Apple Silicon) - placeholder, not yet implemented

## Options

- `--text`, `-t` - Text description of audio to generate (required)
- `--output`, `-o` - Path to save output audio file (required)
- `--voice` - Voice/speaker ID (default: "default")
- `--speed` - Playback speed multiplier (default: 1.0, range: 0.5-2.0)
- `--seed`, `-s` - Random seed for reproducibility
- `--verbose` - Enable verbose logging
- `--device` - Device to use for generation
