from typing import Any

from pydantic import BaseModel, Field


class Text2SpeechRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    voice_id: str | None = Field(None, description="Voice ID or reference audio path")
    emotion: str | None = Field(None, description="Emotion tag (e.g. 'happy')")
    tone: str | None = Field(None, description="Tone tag (e.g. 'whispering')")
    effect: str | None = Field(None, description="Effect tag (e.g. 'laughing')")
    reference_audio: str | None = Field(None, description="Path to reference audio for voice cloning")
    reference_text: str | None = Field(None, description="Transcript of the reference audio")


class Text2SpeechResponse(BaseModel):
    output_path: str = Field(..., description="Path where the output audio is saved")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
