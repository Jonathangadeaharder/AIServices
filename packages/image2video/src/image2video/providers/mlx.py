import os
import random

from aiservices_core.providers import BaseProvider

from ..ltx_frames import normalize_ltx_frame_count
from ..models import Image2VideoRequest, Image2VideoResponse

_DEFAULT_MODEL_DIR = "dgrauet/ltx-2.3-mlx-q8"


class MLXProvider(BaseProvider):
    """Local image-to-video provider using LTX 2.3 MLX (two-stage I2V)."""

    def __init__(
        self,
        model_dir: str | None = None,
        stage1_steps: int = 15,
        stage2_steps: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_dir = model_dir or os.environ.get(
            "IMAGE2VIDEO_MODEL_DIR",
            str(_DEFAULT_MODEL_DIR),
        )
        self._stage1_steps = stage1_steps
        self._stage2_steps = stage2_steps
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        from ltx_pipelines_mlx import TwoStagePipeline

        self._pipeline = TwoStagePipeline(model_dir=self._model_dir, low_memory=True)

    def generate(
        self, request: Image2VideoRequest, output_path: str | None = None
    ) -> Image2VideoResponse:
        self._load_pipeline()
        if self._pipeline is None:
            raise RuntimeError("Pipeline failed to load")

        if output_path is None:
            output_path = "output.mp4"

        effective_seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        num_frames = normalize_ltx_frame_count(
            request.num_frames / request.fps,
            request.fps,
        )

        # Honour the caller's requested step count.  When num_inference_steps is
        # provided, distribute them across both pipeline stages proportionally
        # (80 % stage-1, 20 % stage-2), with a minimum of 1 for stage 2.
        if request.num_inference_steps is not None:
            total = request.num_inference_steps
            stage2 = max(1, round(total * 0.2))
            stage1 = total - stage2
        else:
            stage1 = self._stage1_steps
            stage2 = self._stage2_steps

        self._pipeline.generate_and_save(
            prompt=request.prompt,
            output_path=output_path,
            image=request.image_path,
            height=request.height,
            width=request.width,
            num_frames=num_frames,
            seed=effective_seed,
            stage1_steps=stage1,
            stage2_steps=stage2,
        )

        return Image2VideoResponse(
            output_path=output_path,
            metadata={
                "provider": "mlx",
                "pipeline": "ltx-two-stage",
                "model_dir": self._model_dir,
                "seed": effective_seed,
                "num_frames": num_frames,
                "fps": request.fps,
            },
        )
