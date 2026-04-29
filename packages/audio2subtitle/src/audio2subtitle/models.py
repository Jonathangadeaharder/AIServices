from typing import Any

from pydantic import BaseModel, Field


class Audio2SubtitleRequest(BaseModel):
    audio_path: str = Field(..., description="Path to the input audio file")
    language: str | None = Field(None, description="Language of the audio (auto-detect if None)")
    output_format: str = Field("srt", description="Output format: srt or vtt")
    model_name: str = Field(
        "mlx-community/whisper-large-v3-turbo",
        description="Whisper model name",
    )
    word_timestamps: bool = Field(True, description="Enable word-level timestamps")


class SubtitleEntry(BaseModel):
    index: int = Field(..., description="Subtitle entry index (1-based)")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Subtitle text")


class Audio2SubtitleResponse(BaseModel):
    output_path: str = Field(..., description="Path to the output subtitle file")
    entries: list[SubtitleEntry] = Field(default_factory=list, description="Subtitle entries")
    language: str | None = Field(None, description="Detected language")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
