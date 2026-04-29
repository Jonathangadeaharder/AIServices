from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SubtitleEntry(BaseModel):
    index: int = Field(..., ge=1, description="Subtitle entry index (1-based)")
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., gt=0, description="End time in seconds")
    text: str = Field(..., description="Subtitle text content")

    @model_validator(mode="after")
    def validate_time_order(self) -> "SubtitleEntry":
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        return self


class Video2SubtitleRequest(BaseModel):
    video_path: str = Field(..., description="Path to the input video file")
    language: str | None = Field(None, description="Language code (e.g. 'en')")
    output_format: Literal["srt", "vtt"] = Field("srt", description="Output format")
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
