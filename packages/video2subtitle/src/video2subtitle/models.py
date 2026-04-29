from typing import Any

from pydantic import BaseModel, Field


class SubtitleEntry(BaseModel):
    index: int = Field(..., description="Subtitle entry index (1-based)")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Subtitle text content")


class Video2SubtitleRequest(BaseModel):
    video_path: str = Field(..., description="Path to the input video file")
    language: str | None = Field(None, description="Language code (e.g. 'en')")
    output_format: str = Field("srt", description="Output format: 'srt' or 'vtt'")
    model_name: str = Field(
        "mlx-community/whisper-large-v3-turbo",
        description="Whisper model name",
    )
    word_timestamps: bool = Field(True, description="Enable word-level timestamps")


class Video2SubtitleResponse(BaseModel):
    output_path: str = Field(..., description="Path to the written subtitle file")
    entries: list[SubtitleEntry] = Field(default_factory=list, description="Subtitle entries")
    language: str | None = Field(None, description="Detected language")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
