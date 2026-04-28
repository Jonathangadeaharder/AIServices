from typing import Any

from pydantic import BaseModel, Field


class Speech2TextRequest(BaseModel):
    """Request model for speech-to-text transcription."""

    audio_path: str = Field(..., description="Path to the input audio file")
    model_name: str = Field(
        "mlx-community/whisper-large-v3-mlx", description="Whisper model name"
    )
    language: str | None = Field(None, description="Language of the audio")


class Speech2TextResponse(BaseModel):
    """Response model for speech-to-text transcription."""

    text: str = Field(..., description="Transcribed text")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Transcription metadata"
    )
