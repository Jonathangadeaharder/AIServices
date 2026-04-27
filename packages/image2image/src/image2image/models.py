from typing import Any

from pydantic import BaseModel, Field


class Image2ImageRequest(BaseModel):
    """Request model for image-to-image generation."""

    image_path: str = Field(..., description="Path or URL to the input image")
    prompt: str = Field(..., description="Text prompt for the generation")
    negative_prompt: str | None = Field(None, description="Negative text prompt")
    strength: float = Field(0.5, ge=0.0, le=1.0, description="Transformation strength")
    guidance_scale: float = Field(7.5, description="Guidance scale (CFG)")
    num_inference_steps: int = Field(50, description="Number of denoising steps")
    seed: int | None = Field(None, description="Random seed")


class Image2ImageResponse(BaseModel):
    """Response model for image-to-image generation."""

    output_path: str = Field(..., description="Path where the output image is saved")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
