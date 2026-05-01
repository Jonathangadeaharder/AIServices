#!/usr/bin/env python3
"""
Pipeline: German video → German SRT → Spanish SRT

Usage:
  python scripts/pipe_de2es.py

Steps: ffmpeg → audio2subtitle → subtitle-translate
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


def process_video(video_path: str) -> None:
    video_name = Path(video_path).stem
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    srt_path = OUTPUT_DIR / f"{video_name}.es.srt"

    print(f"\n{'='*60}")
    print(f"  {video_name}")
    print(f"{'='*60}")

    # Step 1: video → audio (ffmpeg)
    print("  [1/3] Extracting audio...")
    audio_path = tempfile.mktemp(suffix=".wav")
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"],
        capture_output=True, timeout=600, check=True,
    )

    # Step 2: audio → German SRT (audio2subtitle)
    print("  [2/3] Transcribing DE...")
    de_srt = tempfile.mktemp(suffix=".srt")
    subprocess.run(
        [sys.executable, "-m", "audio2subtitle.cli",
         "--audio", audio_path, "--output", de_srt, "--language", "de", "--format", "srt"],
        timeout=900, check=True,
    )
    Path(audio_path).unlink(missing_ok=True)

    # Step 3: German SRT → Spanish SRT (subtitle-translate)
    print("  [3/3] Translating DE→ES...")
    subprocess.run(
        [sys.executable, "-m", "subtitle_translate.cli",
         "--input", de_srt, "--output", str(srt_path), "--to", "es", "--ct2-model", CT2_MODEL],
        timeout=300, check=True,
    )
    Path(de_srt).unlink(missing_ok=True)

    print(f"  ✓ → {srt_path}")


def main():
    print("Pipeline: video → audio → DE SRT → ES SRT")
    for video in VIDEO_FILES:
        if not Path(video).exists():
            print(f"  ⚠ Not found: {video}")
            continue
        process_video(video)
    print(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
