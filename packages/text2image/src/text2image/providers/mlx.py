import random
from pathlib import Path

import numpy as np
from aiservices_core.providers import BaseProvider
from PIL import Image

from ..models import Text2ImageRequest, Text2ImageResponse


class MLXProvider(BaseProvider):
    DEFAULT_MODEL = "flux-schnell"
    DEFAULT_STEPS = 4

    def __init__(
        self,
        model_name: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name or self.DEFAULT_MODEL
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        try:
            from image2image.flux_mlx import FluxPipeline  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "flux_mlx is not installed. Install image2image with [flux] extra."
            ) from e

        self._pipeline = FluxPipeline(self.model_name)

    def generate(
        self, request: Text2ImageRequest, output_path: str | None = None
    ) -> Text2ImageResponse:
        self._load_pipeline()
        if self._pipeline is None:
            raise RuntimeError("Pipeline failed to load")

        if output_path is None:
            output_path = "output.png"

        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)

        latent_h = request.height // 16
        latent_w = request.width // 16

        images = self._pipeline.generate_images(
            text=request.prompt,
            num_steps=request.num_inference_steps or self.DEFAULT_STEPS,
            guidance=request.guidance_scale,
            latent_size=(latent_h, latent_w),
            seed=seed,
        )

        images_np = np.array(images[0])
        images_np = (images_np * 255).astype(np.uint8)
        img = Image.fromarray(images_np)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

        return Text2ImageResponse(
            output_path=output_path,
            metadata={
                "provider": "mlx",
                "model_name": self.model_name,
                "seed": seed,
            },
        )
