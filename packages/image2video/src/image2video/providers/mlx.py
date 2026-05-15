import os
import random
from pathlib import Path

from aiservices_core.providers import BaseProvider

from ..models import Image2VideoRequest, Image2VideoResponse

_DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "models" / "ltx-2.3" / "q8"
)


class MLXProvider(BaseProvider):
    """Local image-to-video provider using MLX (Apple Silicon).

    Uses the ltx-2-mlx implementation and local weights.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_dir = model_dir or os.environ.get(
            "IMAGE2VIDEO_MODEL_DIR",
            str(_DEFAULT_MODEL_DIR),
        )
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        from ltx_pipelines_mlx.ti2vid_one_stage import (
            TI2VidOneStagePipeline,
        )

        self._pipeline = TI2VidOneStagePipeline(model_dir=self._model_dir)

    def generate(
        self, request: Image2VideoRequest, output_path: str | None = None
    ) -> Image2VideoResponse:
        self._load_pipeline()
        if self._pipeline is None:
            raise RuntimeError("Pipeline failed to load")

        if output_path is None:
            output_path = "output.mp4"

        effective_seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)

        # Use generate_and_save which handles decoding and saving internally
        self._pipeline.generate_and_save(
            prompt=request.prompt,
            output_path=output_path,
            image=request.image_path,
            height=request.height,
            width=request.width,
            num_frames=request.num_frames,
            frame_rate=request.fps,
            seed=effective_seed,
            num_steps=request.num_inference_steps,
        )

        return Image2VideoResponse(
            output_path=output_path,
            metadata={
                "provider": "mlx",
                "model_dir": self._model_dir,
                "seed": effective_seed,
            },
        )
