from pathlib import Path

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

        # Optional: vocabulary filtering
        if request.vocab_filter_path:
            from ..filter import VocabFilter
            import spacy

            vocab_set = VocabFilter.load_vocab(request.vocab_filter_path, request.vocab_levels)
            nlp = spacy.load("de_core_news_lg", disable=["parser", "ner"])
            entries = VocabFilter.filter_segments(entries, nlp, vocab_set)
            for i, entry in enumerate(entries, 1):
                entry.index = i

        # Optional: translation
        if request.translate_to and entries:
            from ..translator import MarianTranslator

            translator = MarianTranslator(request.translation_model, request.ct2_model_path)
            texts = [e.text for e in entries]
            translated = translator.translate_batch(texts)
            for entry, trans in zip(entries, translated):
                entry.translated_text = trans

        # Write output
        fmt = request.output_format
        if fmt not in {"srt", "vtt"}:
            raise ValueError(f"Unsupported output format: {fmt}")
        output = output_path or f"output.{fmt}"
        Path(output).parent.mkdir(parents=True, exist_ok=True)

        if fmt == "vtt":
            content = "WEBVTT\n\n"
            for entry in entries:
                start = _format_timestamp(entry.start_time, "vtt")
                end = _format_timestamp(entry.end_time, "vtt")
                text = entry.translated_text or entry.text
                content += f"{start} --> {end}\n{text}\n\n"
        else:
            content = ""
            for entry in entries:
                start = _format_timestamp(entry.start_time, "srt")
                end = _format_timestamp(entry.end_time, "srt")
                text = entry.translated_text or entry.text
                content += f"{entry.index}\n{start} --> {end}\n{text}\n\n"

        with open(output, "w", encoding="utf-8") as f:
            f.write(content)

        return Audio2SubtitleResponse(
            output_path=output,
            entries=entries,
            language=str(result.get("language")) if result.get("language") else None,
            metadata={
                "provider": "mlx-whisper",
                "model": request.model_name,
                "format": fmt,
                "total_entries": len(entries),
                "word_timestamps": request.word_timestamps,
                "filtered": request.vocab_filter_path is not None,
                "translated": request.translate_to is not None,
            },
        )
