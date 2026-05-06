"""Video-to-subtitle example using the Python API.

Extracts audio from video and transcribes it to SRT subtitles using Whisper MLX.

Usage:
    uv run python examples/video2subtitle_example.py

Requires a video file at input.mp4 (or change INPUT_PATH below).
Requires FFmpeg installed on PATH.
"""

from pathlib import Path

from video2subtitle.models import Video2SubtitleRequest
from video2subtitle.providers import registry

INPUT_PATH = Path("input.mp4")
OUTPUT_PATH = Path("output_video2subtitle.srt")


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"Place a video file at {INPUT_PATH} to run this example.")
        return

    request = Video2SubtitleRequest(
        video_path=str(INPUT_PATH),
        language="en",
        output_format="srt",
    )

    provider = registry.get("video2subtitle.mlx")
    response = provider.generate(request, output_path=str(OUTPUT_PATH))
    print(f"Generated subtitles saved to {response.output_path}")
    print(f"Detected language: {response.language}")
    print(f"Subtitle entries: {len(response.entries)}")


if __name__ == "__main__":
    main()
