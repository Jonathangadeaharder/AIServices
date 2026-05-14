from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class AudioCategory(StrEnum):
    music = "music"
    sfx = "sfx"
    ambient = "ambient"
    speech = "speech"


class Text2AudioRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt describing the audio")
    duration_seconds: float = Field(10.0, gt=0, description="Duration in seconds")
    output_format: Literal["wav", "mp3"] = Field("wav", description="Output format")
    category: AudioCategory = Field(AudioCategory.music, description="Audio category")
    seed: int | None = Field(None, description="Random seed for reproducibility")


class Text2AudioResponse(BaseModel):
    output_path: str = Field(..., description="Path where the output audio is saved")
    duration_seconds: float | None = Field(None, description="Actual duration of generated audio")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
