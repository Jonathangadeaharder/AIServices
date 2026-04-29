from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class AudioCategory(str, Enum):
    MUSIC = "music"
    SFX = "sfx"
    AMBIENT = "ambient"
    SPEECH = "speech"


class Text2AudioRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt describing the audio to generate")
    negative_prompt: str = Field("", description="Negative prompt for what to avoid")
    duration_seconds: float = Field(10.0, ge=0.5, le=300, description="Duration in seconds")
    output_format: Literal["wav", "mp3"] = Field(
        "wav", description="Output audio format (wav, mp3)"
    )
    category: AudioCategory = Field(
        AudioCategory.MUSIC, description="Category of audio to generate"
    )
    seed: int | None = Field(None, description="Random seed for reproducibility")
    model_version: str = Field("meta/musicgen", description="Model version identifier")


class Text2AudioResponse(BaseModel):
    output_path: str = Field(..., description="Path where the output audio is saved")
    duration_seconds: float | None = Field(None, description="Actual duration of generated audio")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
