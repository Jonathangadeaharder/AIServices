# text2audio

Text-to-audio generation pipeline for the AIServices monorepo. Generates non-speech audio: music, sound effects, and ambient textures.

## Installation

```bash
uv sync --all-packages
```

## Usage

### CLI

```bash
text2audio generate --prompt "calm piano music" --output out.wav --category music --duration 10
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

provider = registry.get("text2audio.replicate")
response = provider.generate(request, "output.wav")
```

## Providers

| Name | Description |
|------|-------------|
| `text2audio.replicate` | Replicate cloud API (Meta MusicGen) |

## Environment Variables

- `REPLICATE_API_TOKEN` - Required for the Replicate provider

## Audio Categories

- `music` - Musical compositions
- `sfx` - Sound effects
- `ambient` - Ambient textures and soundscapes
- `speech` - Speech-like audio (non-TTS)
