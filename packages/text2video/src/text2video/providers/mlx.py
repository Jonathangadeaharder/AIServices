from pathlib import Path

from aiservices_core.providers import BaseProvider
from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline

from ..models import Text2VideoRequest, Text2VideoResponse


class MLXProvider(BaseProvider):
    """Local text-to-video provider using MLX (Apple Silicon).
    
    Uses the ltx-2-mlx implementation and local weights.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Default to the internal models directory
        if model_dir is None:
            # Resolve relative to the package if possible, or use absolute for now
            # In a monorepo, it's better to use an absolute path or a well-known root
            base_dir = (
                Path(__file__).parent.parent.parent.parent.parent.parent
                / "models"
                / "ltx-2.3"
                / "q8"
            )
            model_dir = str(base_dir)
        
        self.pipeline = TextToVideoPipeline(model_dir=model_dir)

    def generate(self, request: Text2VideoRequest, output_path: str) -> Text2VideoResponse:
        result_path = self.pipeline.generate_and_save(
            prompt=request.prompt,
            output_path=output_path,
            height=request.height,
            width=request.width,
            num_frames=request.num_frames,
            seed=request.seed if request.seed is not None else 42,
            num_steps=request.num_inference_steps,
        )

        return Text2VideoResponse(
            output_path=str(result_path) if result_path else output_path,
            metadata={
                "provider": "mlx",
                "model_dir": str(self.pipeline.model_dir),
                "seed": request.seed,
            },
        )
