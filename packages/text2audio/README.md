# text2audio

Text-to-audio generation pipeline for the AIServices monorepo. Generates non-speech audio: music, sound effects, and ambient textures.

## Installation

```bash
uv sync --all-packages
```

## Usage

### CLI

```bash
text2audio generate --prompt "calm piano music" --output out.wav --category music --duration 10 --provider <provider-name>
```

### Python

```python
from text2audio.models import Text2AudioRequest, AudioCategory
from text2audio.providers import registry

request = Text2AudioRequest(
    prompt="calm piano music",
    duration_seconds=10.0,
    category=AudioCategory.MUSIC,
)

provider = registry.get("<provider-name>")
response = provider.generate(request, "output.wav")
```

## Providers

No providers are currently registered. This package is a placeholder for future local MLX-based audio generation models (e.g. MusicGen, AudioLDM2).

## Audio Categories

- `music` - Musical compositions
- `sfx` - Sound effects
- `ambient` - Ambient textures and soundscapes
- `speech` - Speech-like audio (non-TTS)
