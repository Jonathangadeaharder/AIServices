from typing import Any
from pydantic import BaseModel, Field


class Text2SpeechRequest(BaseModel):
    """Request model for text-to-speech generation."""
    text: str = Field(..., description="Text to convert to speech")
    reference_audio: str | None = Field(None, description="Path to reference audio for voice cloning")
    reference_text: str | None = Field(None, description="Text of the reference audio")
    language: str = Field("zh", description="Language of the text")


class Text2SpeechResponse(BaseModel):
    """Response model for text-to-speech generation."""
    output_path: str = Field(..., description="Path to the generated audio file")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
