"""Audio-to-subtitle example using the Python API.

Transcribes an audio file to SRT subtitles using Whisper MLX on Apple Silicon.

Usage:
    uv run python examples/audio2subtitle_example.py

Requires an audio file at input.wav (or change INPUT_PATH below).
"""

from pathlib import Path

from audio2subtitle.client import generate

INPUT_PATH = Path("input.wav")
OUTPUT_PATH = Path("output_audio2subtitle.srt")


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"Place an audio file at {INPUT_PATH} to run this example.")
        return

    output = generate(
        audio_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        output_format="srt",
        language="en",
    )
    print(f"Generated subtitles saved to {output}")


if __name__ == "__main__":
    main()
