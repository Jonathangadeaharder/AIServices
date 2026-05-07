"""Video-to-audio example using the Python API.

Extracts the audio track from a video file using FFmpeg.

Usage:
    uv run python examples/video2audio_example.py

Requires a video file at input.mp4 (or change INPUT_PATH below).
Requires FFmpeg installed on PATH.
"""

from pathlib import Path

from video2audio.models import Video2AudioRequest
from video2audio.providers import registry

INPUT_PATH = Path("input.mp4")
OUTPUT_PATH = Path("output_video2audio.wav")


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"Place a video file at {INPUT_PATH} to run this example.")
        return

    request = Video2AudioRequest(
        video_path=str(INPUT_PATH),
        output_format="wav",
        sample_rate=44100,
        mono=True,
    )

    provider = registry.get("video2audio.ffmpeg")
    response = provider.generate(request, output_path=str(OUTPUT_PATH))
    print(f"Extracted audio saved to {response.output_path}")
    if response.duration_seconds:
        print(f"Duration: {response.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
