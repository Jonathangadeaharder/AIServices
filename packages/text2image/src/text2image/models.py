from typing import Any

from pydantic import BaseModel, Field, field_validator


class Text2ImageRequest(BaseModel):
    """Request model for text-to-image generation."""

    prompt: str = Field(..., description="Text prompt for the generation")
    negative_prompt: str | None = Field(None, description="Negative text prompt")
    guidance_scale: float = Field(7.5, gt=0.0, le=30.0, description="Guidance scale (CFG)")
    num_inference_steps: int = Field(50, ge=1, le=150, description="Number of denoising steps")
    seed: int | None = Field(None, ge=0, description="Random seed")
    width: int = Field(1024, description="Width of the generated image")
    height: int = Field(1024, description="Height of the generated image")

    @field_validator("width", "height")
    @classmethod
    def validate_dimensions(cls, v: int, info) -> int:
        field_name = info.field_name
        if v < 512:
            raise ValueError(f"{field_name} must be >= 512")
        if v % 8 != 0:
            raise ValueError(f"{field_name} must be divisible by 8")
        return v


class Text2ImageResponse(BaseModel):
    """Response model for text-to-image generation."""

    output_path: str = Field(..., description="Path where the output image is saved")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
