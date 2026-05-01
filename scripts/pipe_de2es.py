#!/usr/bin/env python3
"""
Pipe pipeline: German video → German SRT → (filtered) → Spanish SRT

Usage:
  python scripts/pipe_de2es.py

Composes: video2audio → audio2subtitle → subtitle-filter → subtitle-translate
"""
import subprocess
import sys
import tempfile
from pathlib import Path

VIDEO_FILES = [
    "/Users/jonathangadeaharder/Downloads/Euphoria.US.S03E01.GERMANForced.720p.WEB.h264-SAUERKRAUT.mp4.mp4",
    "/Users/jonathangadeaharder/Downloads/Euphoria.US.S03E02.GERMAN.72.German.DisneyHD.x264-4SF.mkv.mp4",
    "/Users/jonathangadeaharder/Downloads/Euphoria.US.S03E03.GERMANForced.720p.WEB.h264-SAUERKRAUT.mp4.mp4",
]

CT2_MODEL = str(Path.home() / ".cache" / "aiservices" / "ct2-opus-mt-de-es")
OUTPUT_DIR = Path("/Users/jonathangadeaharder/Downloads/subtitles")


def run(cmd: list[str], input_data: str | None = None, timeout: int = 600) -> str:
    """Run a command, optionally piping stdin. Returns stdout."""
    result = subprocess.run(
        cmd, input=input_data, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}", file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.stdout


def process_video(video_path: str) -> None:
    video_name = Path(video_path).stem
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    srt_path = OUTPUT_DIR / f"{video_name}.es.srt"

    print(f"\n{'='*60}")
    print(f"  {video_name}")
    print(f"{'='*60}")

    # Step 1: video → audio (ffmpeg)
    print("  [1/4] Extracting audio...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"],
        capture_output=True, timeout=600, check=True,
    )

    # Step 2: audio → German SRT (audio2subtitle)
    print("  [2/4] Transcribing...")
    de_srt = run([
        sys.executable, "-m", "audio2subtitle.cli", "transcribe",
        "--audio", audio_path, "--output", "/dev/stdout",
        "--language", "de", "--format", "srt",
    ])
    Path(audio_path).unlink(missing_ok=True)

    # Step 3: German SRT → filtered SRT (subtitle-filter)
    print("  [3/4] Filtering vocabulary...")
    filtered_srt = de_srt  # Skip filtering for now (no vocab CSVs available)
    # To enable: pipe through subtitle-filter
    # filtered_srt = run([
    #     sys.executable, "-m", "subtitle_filter.cli", "filter",
    #     "--vocab", "./IdeaProjects/src/backend/data/",
    #     "--levels", "A1,A2,B1",
    # ], input_data=de_srt)

    # Step 4: filtered SRT → Spanish SRT (subtitle-translate)
    print("  [4/4] Translating DE→ES...")
    es_srt = run([
        sys.executable, "-m", "subtitle_translate.cli", "translate",
        "--to", "es", "--ct2-model", CT2_MODEL,
    ], input_data=filtered_srt)

    # Write output
    srt_path.write_text(es_srt, encoding="utf-8")
    print(f"  ✓ → {srt_path}")


def main():
    print("Pipe: video → audio → SRT → translate → Spanish SRT")
    for video in VIDEO_FILES:
        if not Path(video).exists():
            print(f"  ⚠ Not found: {video}")
            continue
        process_video(video)
    print(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
