"""Video generation provider using ltx-pipelines-mlx.

Implements the aiservices_core BaseProvider pattern for video generation.
"""

from __future__ import annotations

from typing import Any

from aiservices_core.providers import BaseProvider

from ltx_pipelines_mlx.api import (
    VideoGenerationResult,
    audio_to_video,
    image_to_video,
    text_to_video,
)


class LtxVideoProvider(BaseProvider):
    """Local video generation provider using LTX-2.3 on MLX.

    Supports text-to-video, image-to-video, and audio-to-video generation
    on Apple Silicon using the MLX framework.

    Usage:
        from aiservices_core.providers import registry
        from ltx_pipelines_mlx.providers import LtxVideoProvider

        # Register the provider
        registry.register("video.ltx", LtxVideoProvider)

        # Get provider instance
        provider = registry.get("video.ltx")

        # Generate video
        result = provider.generate(
            prompt="a cat walking",
            output_path="output.mp4",
        )
    """

    def __init__(
        self,
        model: str | None = None,
        gemma: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the LTX video provider.

        Args:
            model: Model weights (HuggingFace repo ID or local path).
                   Defaults to "dgrauet/ltx-2.3-mlx-q8".
            gemma: Gemma model for text encoding.
                   Defaults to "mlx-community/gemma-3-12b-it-4bit".
            **kwargs: Additional provider configuration.
        """
        super().__init__(**kwargs)
        self.model = model or "dgrauet/ltx-2.3-mlx-q8"
        self.gemma = gemma or "mlx-community/gemma-3-12b-it-4bit"

    def generate(
        self,
        prompt: str,
        output_path: str,
        *,
        mode: str = "t2v",
        image: str | None = None,
        audio: str | None = None,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = -1,
        **kwargs: Any,
    ) -> VideoGenerationResult:
        """Generate a video.

        Args:
            prompt: Text prompt for video generation.
            output_path: Path to save the output video.
            mode: Generation mode - "t2v" (text-to-video), "i2v" (image-to-video),
                  or "a2v" (audio-to-video).
            image: Path to reference image (required for "i2v" mode).
            audio: Path to input audio (required for "a2v" mode).
            height: Video height in pixels.
            width: Video width in pixels.
            num_frames: Number of frames to generate.
            seed: Random seed (-1 for random).
            **kwargs: Additional generation parameters.

        Returns:
            VideoGenerationResult with output path and metadata.

        Raises:
            ValueError: If required parameters are missing for the chosen mode.
        """
        if mode == "t2v":
            return text_to_video(
                prompt=prompt,
                output_path=output_path,
                model=self.model,
                gemma=self.gemma,
                height=height,
                width=width,
                num_frames=num_frames,
                seed=seed,
                **kwargs,
            )
        elif mode == "i2v":
            if image is None:
                raise ValueError("Image path is required for image-to-video mode")
            return image_to_video(
                prompt=prompt,
                image=image,
                output_path=output_path,
                model=self.model,
                gemma=self.gemma,
                height=height,
                width=width,
                num_frames=num_frames,
                seed=seed,
                **kwargs,
            )
        elif mode == "a2v":
            if audio is None:
                raise ValueError("Audio path is required for audio-to-video mode")
            return audio_to_video(
                prompt=prompt,
                audio=audio,
                output_path=output_path,
                model=self.model,
                gemma=self.gemma,
                height=height,
                width=width,
                num_frames=num_frames,
                seed=seed,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 't2v', 'i2v', or 'a2v'.")
