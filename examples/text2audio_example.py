"""Text-to-audio example using the Python API.

Generates non-speech audio (music, SFX, ambient) from a text prompt.
Currently a placeholder — no providers are registered yet.

Usage:
    uv run python examples/text2audio_example.py
"""

from pathlib import Path

from text2audio.models import AudioCategory, Text2AudioRequest
from text2audio.providers import registry

OUTPUT_PATH = Path("output_text2audio.wav")


def main() -> None:
    request = Text2AudioRequest(
        prompt="calm piano music with soft ambient pads",
        duration_seconds=10.0,
        category=AudioCategory.MUSIC,
    )

    try:
        provider = registry.get("text2audio.mlx")
        response = provider.generate(request, output_path=str(OUTPUT_PATH))
        print(f"Generated audio saved to {response.output_path}")
    except ValueError as e:
        print(f"No provider available: {e}")
        print("text2audio is a placeholder — providers will be added in a future release.")


if __name__ == "__main__":
    main()
