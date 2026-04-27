from typing import Any

from pydantic import BaseModel, Field


class Text2ImageRequest(BaseModel):
    """Request model for text-to-image generation."""

    prompt: str = Field(..., description="Text prompt for the generation")
    negative_prompt: str | None = Field(None, description="Negative text prompt")
    guidance_scale: float = Field(7.5, description="Guidance scale (CFG)")
    num_inference_steps: int = Field(50, description="Number of denoising steps")
    seed: int | None = Field(None, description="Random seed")
    width: int = Field(1024, description="Width of the generated image")
    height: int = Field(1024, description="Height of the generated image")


class Text2ImageResponse(BaseModel):
    """Response model for text-to-image generation."""

    output_path: str = Field(..., description="Path where the output image is saved")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
