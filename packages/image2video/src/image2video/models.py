from typing import Any

from pydantic import BaseModel, Field, field_validator


class Image2VideoRequest(BaseModel):
    """Request model for image-to-video generation."""

    image_path: str = Field(..., description="Path to the input image")
    prompt: str = Field(..., description="Text prompt for video generation")
    negative_prompt: str = Field(
        "overexposed, static, blurry, worst quality, low quality, ugly, deformed",
        description="Negative text prompt",
    )
    width: int = Field(640, description="Width of the video frames")
    height: int = Field(640, description="Height of the video frames")
    num_frames: int = Field(81, ge=1, le=257, description="Number of frames")
    num_inference_steps: int = Field(4, ge=1, le=100, description="Number of denoising steps")
    seed: int | None = Field(None, description="Random seed")
    fps: int = Field(16, ge=1, le=60, description="Frames per second for output")

    @field_validator("width", "height")
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        if v % 8 != 0:
            raise ValueError(f"Dimension must be divisible by 8, got {v}")
        if v < 64 or v > 2048:
            raise ValueError(f"Dimension must be 64-2048, got {v}")
        return v


class Image2VideoResponse(BaseModel):
    """Response model for image-to-video generation."""

    output_path: str = Field(..., description="Path where the output video/frames are saved")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
