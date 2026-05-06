"""Text-to-speech example using the Python API.

Synthesizes speech from text using Fish Speech on Apple Silicon.

Usage:
    uv run python examples/text2speech_example.py
"""

from pathlib import Path

from text2speech.client import generate

OUTPUT_PATH = Path("output_text2speech.wav")


def main() -> None:
    output = generate(
        text="Hello, how can I help you today? This is a demo of Fish Speech text-to-speech.",
        output_path=OUTPUT_PATH,
    )
    print(f"Generated speech saved to {output}")


if __name__ == "__main__":
    main()
