import random
from pathlib import Path

import mlx.core as mx
from aiservices_core.providers import BaseProvider
from PIL import Image

from ..models import Image2ImageRequest, Image2ImageResponse


class MLXProvider(BaseProvider):
    """Local image-to-image provider using MLX.

    Runs FLUX.2-klein-9B inference on Apple Silicon via diffusers with MLX backend.
    Downloads the model from HuggingFace on first use.
    """

    DEFAULT_MODEL = "mlx-community/FLUX.2-klein-9B"

    def __init__(
        self,
        model_id: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_id = model_id or self.DEFAULT_MODEL
        self._pipe = None

    def _load_pipeline(self):
        """Lazy-load the FLUX pipeline on first use."""
        if self._pipe is not None:
            return

        import torch
        from diffusers import Flux2KleinPipeline

        self._pipe = Flux2KleinPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        self._pipe.to("mps")

    def generate(self, request: Image2ImageRequest, output_path: str) -> Image2ImageResponse:
        self._load_pipeline()

        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)

        # Load the input image
        input_path = Path(request.image_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {request.image_path}")

        input_image = Image.open(input_path).convert("RGB")

        import torch

        generator = torch.Generator(device="mps").manual_seed(seed)

        # Run image-to-image generation
        result = self._pipe(
            prompt=request.prompt,
            image=input_image,
            strength=request.strength,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            generator=generator,
        )

        # Save output
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(output_path)

        return Image2ImageResponse(
            output_path=output_path,
            metadata={
                "provider": "mlx",
                "model_id": self.model_id,
                "seed": seed,
            },
        )
