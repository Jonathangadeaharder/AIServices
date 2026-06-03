"""Audio transcription abstraction.

Provides:
- BaseTranscriber ABC — callers should depend on this, not on concrete providers
- MLXWhisperTranscriber — local MLX-accelerated Whisper (Apple Silicon)
- FasterWhisperTranscriber — CPU-friendly Whisper via faster-whisper (CTranslate2)
- create_transcriber() factory — chooses provider via env var
- Convenience: transcribe(), transcribe_to_srt()

Callers import from aiservices, never from mlx_whisper / faster_whisper directly.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

try:
    from structlog import get_logger
except ImportError:

    class Log:
        def info(self, *a, **k):
            pass
        def warning(self, *a, **k):
            pass

    def get_logger(n):
        return Log()


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """A transcribed segment with timing."""
    start: float = 0.0
    end: float = 0.0
    text: str = ""


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    text: str = ""
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    language_probability: float = 1.0


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class BaseTranscriber(ABC):
    """Abstract transcriber — all providers implement this.
    
    Callers depend on this interface, not concrete providers.
    """

    @abstractmethod
    def transcribe(
        self, audio_path: str, *, language: str | None = None
    ) -> TranscriptionResult:
        """Transcribe a single audio file."""
        ...

    def transcribe_to_srt(
        self, audio_path: str, output_path: str | Path, *, language: str | None = None
    ) -> Path:
        """Transcribe and write SRT subtitle file."""
        result = self.transcribe(audio_path, language=language)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for i, seg in enumerate(result.segments, 1):
            start = _fmt_time(seg.start)
            end = _fmt_time(seg.end)
            lines += [str(i), f"{start} --> {end}", seg.text.strip(), ""]

        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def transcribe_batch(
        self, audio_paths: list[str], *, language: str | None = None
    ) -> list[TranscriptionResult]:
        """Transcribe multiple audio files."""
        return [self.transcribe(p, language=language) for p in audio_paths]


# ---------------------------------------------------------------------------
# Concrete providers
# ---------------------------------------------------------------------------


class MLXWhisperTranscriber(BaseTranscriber):
    """Local MLX-accelerated Whisper (Apple Silicon only).
    
    Uses mlx-whisper. Falls back gracefully when not installed.
    Model: mlx-community/whisper-large-v3-turbo
    """

    MODEL = "mlx-community/whisper-large-v3-turbo"

    def __init__(self, model: str = MODEL):
        self._model = model

    def transcribe(
        self, audio_path: str, *, language: str | None = None
    ) -> TranscriptionResult:
        try:
            import mlx_whisper  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "mlx-whisper not installed (Apple Silicon required)"
            ) from e

        logger.info(
            "mlx_transcribe_start",
            path=audio_path,
            model=self._model,
            language=language,
        )
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=self._model,
            language=language,
            word_timestamps=True,
        )

        raw_text = result.get("text", "")
        text = raw_text.strip() if isinstance(raw_text, str) else " ".join(raw_text).strip()
        segments_raw = result.get("segments", [])

        segments = [
            Segment(start=s.get("start", 0), end=s.get("end", 0), text=s.get("text", ""))
            for s in segments_raw
        ]

        detected_lang = result.get("language", language)

        logger.info(
            "mlx_transcribe_done",
            path=audio_path,
            chars=len(text),
            segments=len(segments),
            language=detected_lang,
        )
        return TranscriptionResult(
            text=text,
            segments=segments,
            language=detected_lang,
        )


class FasterWhisperTranscriber(BaseTranscriber):
    """CPU-friendly Whisper via faster-whisper (CTranslate2 backend).
    
    Falls back gracefully when faster-whisper not installed.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        device: str | None = None,
        compute_type: str = "float32",
    ):
        self._model_size = model_size
        self._device = device or os.getenv("WHISPER_DEVICE", "cpu")
        self._compute_type = compute_type
        self._model = None  # lazy init

    def _get_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore[import]
            except ImportError as e:
                raise ImportError(
                    "faster-whisper not installed (pip install faster-whisper)"
                ) from e
            logger.info(
                "faster_whisper_load",
                model=self._model_size,
                device=self._device,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    def transcribe(
        self, audio_path: str, *, language: str | None = None
    ) -> TranscriptionResult:
        model = self._get_model()
        logger.info("faster_transcribe_start", path=audio_path, language=language)

        segments, info = model.transcribe(audio_path, language=language, beam_size=5)

        result_segments: list[Segment] = []
        text_parts: list[str] = []
        for s in segments:
            result_segments.append(Segment(start=s.start, end=s.end, text=s.text))
            text_parts.append(s.text)

        return TranscriptionResult(
            text=" ".join(text_parts),
            segments=result_segments,
            language=info.language,
            language_probability=info.language_probability,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_TRANSCRIBER_PROVIDERS: dict[str, type[BaseTranscriber]] = {
    "mlx": MLXWhisperTranscriber,
    "faster-whisper": FasterWhisperTranscriber,
}

_WHISPER_BACKEND = os.getenv("WHISPER_BACKEND", "auto")


def create_transcriber(provider: str | None = None) -> BaseTranscriber:
    """Create a transcriber instance.

    Args:
        provider: One of "mlx", "faster-whisper", "auto" (default).
            Auto tries MLX first, falls back to faster-whisper.

    Returns:
        A BaseTranscriber instance.

    Raises:
        ImportError: If no provider available.
    """
    name = provider or _WHISPER_BACKEND

    if name == "auto":
        try:
            import mlx_whisper  # type: ignore[import]  # noqa: F401
            return MLXWhisperTranscriber()
        except ImportError:
            pass
        try:
            from faster_whisper import WhisperModel  # type: ignore[import]  # noqa: F401
            return FasterWhisperTranscriber()
        except ImportError:
            pass
        raise ImportError(
            "No whisper backend available. "
            "Install mlx-whisper (Apple Silicon) or faster-whisper (CPU)."
        )

    if name == "mlx":
        return MLXWhisperTranscriber()
    if name in ("faster-whisper", "cpu"):
        return FasterWhisperTranscriber()

    raise ValueError(f"Unknown transcriber provider: {name!r}")


# ---------------------------------------------------------------------------
# Convenience functions (backward compatible)
# ---------------------------------------------------------------------------


def transcribe(
    audio_path: str | Path,
    *,
    language: str | None = None,
    provider: str | None = None,
) -> TranscriptionResult:
    """Transcribe audio.

    Args:
        audio_path: Path to audio file
        language: Optional language code (e.g. "en")
        provider: Backend provider ("mlx", "faster-whisper", "auto")

    Returns:
        TranscriptionResult with text, segments, language
    """
    transcriber = create_transcriber(provider)
    return transcriber.transcribe(str(audio_path), language=language)


def transcribe_to_srt(
    audio_path: str | Path,
    output_path: str | Path,
    *,
    language: str | None = None,
    provider: str | None = None,
) -> Path:
    """Transcribe audio and write SRT subtitle file.

    Args:
        audio_path: Path to audio file
        output_path: Output SRT file path
        language: Optional language code
        provider: Backend provider

    Returns:
        Path to written SRT file
    """
    transcriber = create_transcriber(provider)
    return transcriber.transcribe_to_srt(str(audio_path), output_path, language=language)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
