from typing import Any

from pydantic import BaseModel, Field


class Text2AudioRequest(BaseModel):
    text: str = Field(..., description="Text description of audio to generate")
    voice: str = Field("default", description="Voice/speaker ID")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Playback speed multiplier")
    seed: int | None = Field(None, description="Random seed for reproducibility")


class Text2AudioResponse(BaseModel):
    output_path: str = Field(..., description="Path where the output audio is saved")
    duration_seconds: float | None = Field(None, description="Actual duration of generated audio")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
