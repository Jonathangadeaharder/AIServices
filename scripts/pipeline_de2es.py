#!/usr/bin/env python3
"""
Standalone pipeline: German video → Spanish subtitles.
Uses: ffmpeg (audio extraction) → mlx-whisper (transcription) → CTranslate2 MarianMT (translation) → SRT
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

VIDEO_FILES = [
    "/Users/jonathangadeaharder/Downloads/Euphoria.US.S03E01.GERMANForced.720p.WEB.h264-SAUERKRAUT.mp4.mp4",
    "/Users/jonathangadeaharder/Downloads/Euphoria.US.S03E02.GERMAN.72.German.DisneyHD.x264-4SF.mkv.mp4",
    "/Users/jonathangadeaharder/Downloads/Euphoria.US.S03E03.GERMANForced.720p.WEB.h264-SAUERKRAUT.mp4.mp4",
]

WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
MT_MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-de-es"
CT2_MODEL_DIR = Path.home() / ".cache" / "aiservices" / "ct2-opus-mt-de-es"
OUTPUT_DIR = Path("/Users/jonathangadeaharder/Downloads/subtitles")


def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract audio from video using ffmpeg."""
    print(f"  Extracting audio...")
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"],
        capture_output=True, timeout=600, check=True,
    )


def transcribe(audio_path: str) -> list[dict]:
    """Transcribe audio using mlx-whisper."""
    import mlx_whisper

    print(f"  Transcribing with mlx-whisper...")
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=WHISPER_MODEL,
        language="de",
        word_timestamps=True,
    )
    segments = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if text:
            segments.append({"start": seg["start"], "end": seg["end"], "text": text})
    return segments


def convert_marianmt_to_ct2() -> None:
    """Convert MarianMT model to CTranslate2 format if not already done."""
    if (CT2_MODEL_DIR / "model.bin").exists():
        print(f"  CT2 model already exists at {CT2_MODEL_DIR}")
        return

    print(f"  Converting {MT_MODEL_NAME} to CTranslate2 format...")
    CT2_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "ctranslate2.converters.transformers",
         "--model", MT_MODEL_NAME, "--output_dir", str(CT2_MODEL_DIR)],
        check=True, timeout=300,
    )


def translate_segments(segments: list[dict]) -> list[str]:
    """Translate German segments to Spanish using CTranslate2."""
    import ctranslate2
    import transformers

    if not segments:
        return []

    print(f"  Translating {len(segments)} segments DE→ES...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MT_MODEL_NAME)
    translator = ctranslate2.Translator(str(CT2_MODEL_DIR))

    translations = []
    batch_size = 32
    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        source_tokens = [
            tokenizer.convert_ids_to_tokens(tokenizer.encode(s["text"]))
            for s in batch
        ]
        results = translator.translate_batch(source_tokens)
        for result in results:
            target_ids = tokenizer.convert_tokens_to_ids(result.hypotheses[0])
            translations.append(tokenizer.decode(target_ids, skip_special_tokens=True))
    return translations


def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], translations: list[str], output_path: str) -> None:
    """Write translated SRT file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (seg, trans) in enumerate(zip(segments, translations), 1):
            start = format_srt_time(seg["start"])
            end = format_srt_time(seg["end"])
            f.write(f"{i}\n{start} --> {end}\n{trans}\n\n")


def process_video(video_path: str) -> None:
    """Process a single video file through the full pipeline."""
    video_name = Path(video_path).stem
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    srt_path = str(OUTPUT_DIR / f"{video_name}.es.srt")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        t0 = time.time()

        # Step 1: Extract audio
        extract_audio(video_path, audio_path)

        # Step 2: Transcribe
        segments = transcribe(audio_path)
        print(f"  Got {len(segments)} segments")

        # Step 3: Translate
        translations = translate_segments(segments)

        # Step 4: Write SRT
        write_srt(segments, translations, srt_path)

        elapsed = time.time() - t0
        print(f"  ✓ Done in {elapsed:.1f}s → {srt_path}")

    finally:
        Path(audio_path).unlink(missing_ok=True)


def main():
    print("Pipeline: German Video → Spanish Subtitles")
    print(f"Whisper model: {WHISPER_MODEL}")
    print(f"Translation model: {MT_MODEL_NAME}")

    # Convert MarianMT to CT2 format
    convert_marianmt_to_ct2()

    # Process each video
    for video_path in VIDEO_FILES:
        if not Path(video_path).exists():
            print(f"  ⚠ File not found: {video_path}")
            continue
        process_video(video_path)

    print(f"\nAll done. Subtitles saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
