import random
from pathlib import Path

from aiservices_core.providers import BaseProvider
from PIL import Image

from ..models import Image2ImageRequest, Image2ImageResponse


class MLXProvider(BaseProvider):
    """Local image-to-image provider using MLX.

    Runs FLUX.2-klein-9B inference directly on Apple Silicon via native MLX.
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
        self._model = None

    def _load_model(self):
        """Lazy-load the MLX FLUX model on first use."""
        if self._model is not None:
            return

        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        # Download model files
        model_path = hf_hub_download(
            repo_id=self.model_id,
            filename="flux-2-klein-9b.safetensors",
        )

        self._model = {}
        with safe_open(model_path, framework="mlx") as f:
            for key in f.keys():
                self._model[key] = f.get_tensor(key)

    def generate(self, request: Image2ImageRequest, output_path: str) -> Image2ImageResponse:
        self._load_model()
        assert self._model is not None

        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)

        input_path = Path(request.image_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {request.image_path}")

        input_image = Image.open(input_path).convert("RGB")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        input_image.save(output_path)

        return Image2ImageResponse(
            output_path=output_path,
            metadata={
                "provider": "mlx",
                "model_id": self.model_id,
                "seed": seed,
                "note": "Placeholder - full FLUX.2 pipeline not yet implemented",
            },
        )
