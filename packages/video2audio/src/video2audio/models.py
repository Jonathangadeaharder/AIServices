from typing import Any, Literal

from pydantic import BaseModel, Field


class Video2AudioRequest(BaseModel):
    """Request model for video-to-audio extraction."""

    video_path: str = Field(..., min_length=1, description="Path to input video file")
    output_format: Literal["wav", "mp3", "aac"] = Field("wav", description="Output audio format")
    sample_rate: int = Field(44100, gt=0, le=192000, description="Output sample rate in Hz")
    mono: bool = Field(True, description="Convert to mono")


class Video2AudioResponse(BaseModel):
    """Response model for video-to-audio extraction."""

    output_path: str = Field(..., description="Path to the output audio file")
    duration_seconds: float | None = Field(
        None, description="Duration of extracted audio in seconds"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extraction metadata")
