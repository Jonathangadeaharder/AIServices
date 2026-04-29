from pathlib import Path

from aiservices_core.providers import BaseProvider

from ..models import Image2VideoRequest, Image2VideoResponse


class MLXProvider(BaseProvider):
    """Local image-to-video provider using MLX (Apple Silicon).

    Uses the ltx-2-mlx implementation and local weights.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        **kwargs,
    ):
        from ltx_pipelines_mlx.ti2vid_one_stage import ImageToVideoPipeline

        super().__init__(**kwargs)
        if model_dir is None:
            base_dir = (
                Path(__file__).parent.parent.parent.parent.parent.parent
                / "models"
                / "ltx-2.3"
                / "q8"
            )
            model_dir = str(base_dir)

        self.pipeline = ImageToVideoPipeline(model_dir=model_dir)

    def generate(
        self, request: Image2VideoRequest, output_path: str | None = None
    ) -> Image2VideoResponse:
        from PIL import Image

        if output_path is None:
            output_path = "output.mp4"

        effective_seed = request.seed if request.seed is not None else 42
        with Image.open(request.image_path) as image:
            video_latent, _ = self.pipeline.generate_from_image(
                prompt=request.prompt,
                image=image,
                height=request.height,
                width=request.width,
                num_frames=request.num_frames,
                seed=effective_seed,
                num_steps=request.num_inference_steps,
            )

        self.pipeline.save_video(
            video_latent,
            output_path,
            fps=request.fps,
        )

        return Image2VideoResponse(
            output_path=output_path,
            metadata={
                "provider": "mlx",
                "model_dir": str(self.pipeline.model_dir),
                "seed": request.seed,
            },
        )
