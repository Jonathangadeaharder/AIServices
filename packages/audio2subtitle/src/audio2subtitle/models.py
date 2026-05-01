from typing import Any, Literal

from pydantic import BaseModel, Field


class Audio2SubtitleRequest(BaseModel):
    audio_path: str = Field(..., description="Path to the input audio file")
    language: str | None = Field(None, description="Language of the audio (auto-detect if None)")
    output_format: Literal["srt", "vtt"] = Field("srt", description="Output format: srt or vtt")
    model_name: str = Field(
        "mlx-community/whisper-large-v3-turbo",
        description="Whisper model name",
    )
    word_timestamps: bool = Field(True, description="Enable word-level timestamps")
    translate_to: str | None = Field(
        None, description="Target language code for translation (e.g. 'es'). None = no translation."
    )
    translation_model: str = Field(
        "Helsinki-NLP/opus-mt-tc-big-de-es",
        description="HuggingFace translation model name (for tokenizer)",
    )
    ct2_model_path: str = Field(
        "",
        description="Path to CTranslate2-converted model directory.",
    )
    vocab_filter_path: str | None = Field(
        None,
        description="Path to directory containing vocab CSVs (A1_vokabeln.csv, etc.). None = no filtering.",
    )
    vocab_levels: list[str] = Field(
        default_factory=lambda: ["A1", "A2", "B1"],
        description="Vocabulary levels to load for filtering",
    )


class SubtitleEntry(BaseModel):
    index: int = Field(..., description="Subtitle entry index (1-based)")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Subtitle text")
    translated_text: str | None = Field(None, description="Translated subtitle text")


class Audio2SubtitleResponse(BaseModel):
    output_path: str = Field(..., description="Path to the output subtitle file")
    entries: list[SubtitleEntry] = Field(default_factory=list, description="Subtitle entries")
    language: str | None = Field(None, description="Detected language")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
