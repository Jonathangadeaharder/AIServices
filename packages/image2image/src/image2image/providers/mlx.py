import random
from pathlib import Path

from aiservices_core.providers import BaseProvider

from ..models import Image2ImageRequest, Image2ImageResponse


class MLXProvider(BaseProvider):
    """Local image-to-image provider using mflux Flux2KleinEdit on Apple Silicon.

    Fixed to FLUX.2-Klein-9B (mlx-community). Quantize defaults to 8-bit.
    """

    MODEL_NAME = "flux2-klein-9b"
    # mlx-community variant (ungated) statt black-forest-labs (gated)
    MODEL_PATH = "mlx-community/FLUX.2-klein-9B"

    def __init__(self, quantize: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.quantize = quantize
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

        self._model = Flux2KleinEdit(
            model_config=ModelConfig.from_name(model_name=self.MODEL_NAME),
            model_path=self.MODEL_PATH,
            quantize=self.quantize,
        )

    def generate(self, request: Image2ImageRequest, output_path: str) -> Image2ImageResponse:
        self._load_model()

        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        from PIL import Image as PILImage
        from mflux.utils.image_util import ImageUtil

        # Match input dims (rounded down to multiple of 16, capped at 2048 long edge).
        # Default 1024 ruined text on A4 (2480x3509 → 1024 unreadable).
        with PILImage.open(request.image_path) as im:
            iw, ih = im.size
        max_long = 1536
        scale = min(1.0, max_long / max(iw, ih))
        width = max(256, int(iw * scale) // 16 * 16)
        height = max(256, int(ih * scale) // 16 * 16)

        image = self._model.generate_image(
            seed=seed,
            prompt=request.prompt,
            num_inference_steps=request.num_inference_steps,
            width=width,
            height=height,
            guidance=1.0,  # flux2 distilled: guidance must be 1.0
            image_paths=[request.image_path],
            image_strength=request.strength,
        )

        ImageUtil.save_image(image=image, path=output_path)

        return Image2ImageResponse(
            output_path=output_path,
            metadata={
                "provider": "mlx",
                "model": self.MODEL_NAME,
                "quantize": self.quantize,
                "seed": seed,
                "strength": request.strength,
                "steps": request.num_inference_steps,
                "width": width,
                "height": height,
            },
        )
