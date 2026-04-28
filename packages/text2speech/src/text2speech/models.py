from typing import Any

from pydantic import BaseModel, Field


class Text2SpeechRequest(BaseModel):
    """Request model for text-to-speech generation."""

    text: str = Field(..., description="Text to convert to speech")
    voice_id: str | None = Field(None, description="Voice ID or reference audio path")


class Text2SpeechResponse(BaseModel):
    """Response model for text-to-speech generation."""

    output_path: str = Field(..., description="Path where the output audio is saved")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
