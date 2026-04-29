from aiservices_core.providers import BaseProvider

from ..models import Audio2SubtitleRequest, Audio2SubtitleResponse, SubtitleEntry


def _format_timestamp(seconds: float, fmt: str = "srt") -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    if fmt == "vtt":
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class MLXWhisperProvider(BaseProvider):
    """Audio-to-subtitle provider using mlx-whisper (Apple Silicon)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(
        self, request: Audio2SubtitleRequest, output_path: str | None = None
    ) -> Audio2SubtitleResponse:
        try:
            import mlx_whisper

            result = mlx_whisper.transcribe(
                request.audio_path,
                path_or_hf_repo=request.model_name,
                language=request.language,
                word_timestamps=request.word_timestamps,
            )
        except Exception as e:
            raise RuntimeError(f"Subtitle generation failed: {e}") from e

        segments = result.get("segments", [])
        entries: list[SubtitleEntry] = []
        for i, seg in enumerate(segments, start=1):
            text = seg.get("text", "").strip()
            if not text:
                continue
            entries.append(
                SubtitleEntry(
                    index=i,
                    start_time=seg.get("start", 0.0),
                    end_time=seg.get("end", 0.0),
                    text=text,
                )
            )

        fmt = request.output_format
        if fmt not in {"srt", "vtt"}:
            raise ValueError(f"Unsupported output format: {fmt}")
        output = output_path or f"output.{fmt}"

        if fmt == "vtt":
            content = "WEBVTT\n\n"
            for entry in entries:
                start = _format_timestamp(entry.start_time, "vtt")
                end = _format_timestamp(entry.end_time, "vtt")
                content += f"{start} --> {end}\n{entry.text}\n\n"
        else:
            content = ""
            for entry in entries:
                start = _format_timestamp(entry.start_time, "srt")
                end = _format_timestamp(entry.end_time, "srt")
                content += f"{entry.index}\n{start} --> {end}\n{entry.text}\n\n"

        with open(output, "w", encoding="utf-8") as f:
            f.write(content)

        return Audio2SubtitleResponse(
            output_path=output,
            entries=entries,
            language=result.get("language"),
            metadata={
                "provider": "mlx-whisper",
                "model": request.model_name,
                "format": fmt,
                "total_entries": len(entries),
                "word_timestamps": request.word_timestamps,
            },
        )
