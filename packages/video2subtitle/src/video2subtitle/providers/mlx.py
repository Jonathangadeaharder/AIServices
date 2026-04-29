import shutil
import subprocess
import tempfile
from pathlib import Path

from aiservices_core.providers import BaseProvider

from ..models import SubtitleEntry, Video2SubtitleRequest, Video2SubtitleResponse


class MLXProvider(BaseProvider):
    """Video-to-subtitle provider using ffmpeg + mlx-whisper."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(
        self, request: Video2SubtitleRequest, output_path: str | None = None
    ) -> Video2SubtitleResponse:
        if not output_path:
            raise ValueError("output_path is required for video2subtitle")

        audio_path = self._extract_audio(request.video_path)

        try:
            import mlx_whisper

            result = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=request.model_name,
                language=request.language,
                word_timestamps=request.word_timestamps,
            )
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}") from e
        finally:
            Path(audio_path).unlink(missing_ok=True)

        entries = self._segments_to_entries(result.get("segments", []))
        self._write_subtitle_file(entries, output_path, request.output_format)

        return Video2SubtitleResponse(
            output_path=str(output_path),
            entries=entries,
            language=result.get("language"),
            metadata={
                "provider": "video2subtitle.mlx",
                "model": request.model_name,
                "format": request.output_format,
                "word_timestamps": request.word_timestamps,
            },
        )

    def _extract_audio(self, video_path: str) -> str:
        ffmpeg = self._find_ffmpeg()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()

        cmd = [
            ffmpeg,
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            tmp.name,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            Path(tmp.name).unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg audio extraction failed: {e.stderr.decode()}") from e

        return tmp.name

    def _segments_to_entries(self, segments: list[dict]) -> list[SubtitleEntry]:
        entries: list[SubtitleEntry] = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            entries.append(
                SubtitleEntry(
                    index=len(entries) + 1,
                    start_time=seg.get("start", 0.0),
                    end_time=seg.get("end", 0.0),
                    text=text,
                )
            )
        return entries

    def _write_subtitle_file(
        self, entries: list[SubtitleEntry], output_path: str, fmt: str
    ) -> None:
        lines: list[str] = []

        if fmt == "vtt":
            lines.append("WEBVTT")
            lines.append("")
            for entry in entries:
                lines.append(
                    f"{self._format_timestamp(entry.start_time, 'vtt')}"
                    f" --> "
                    f"{self._format_timestamp(entry.end_time, 'vtt')}"
                )
                lines.append(entry.text)
                lines.append("")
        else:
            for entry in entries:
                lines.append(str(entry.index))
                lines.append(
                    f"{self._format_timestamp(entry.start_time, 'srt')}"
                    f" --> "
                    f"{self._format_timestamp(entry.end_time, 'srt')}"
                )
                lines.append(entry.text)
                lines.append("")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _format_timestamp(seconds: float, fmt: str = "srt") -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        sep = "." if fmt == "vtt" else ","
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"

    @staticmethod
    def _find_ffmpeg() -> str:
        path = shutil.which("ffmpeg")
        if not path:
            raise RuntimeError("ffmpeg not found. Install ffmpeg and ensure it is on PATH.")
        return path
