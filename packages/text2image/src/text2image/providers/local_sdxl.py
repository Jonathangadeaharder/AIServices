import torch
from aiservices_core.providers import BaseProvider
from aiservices_core.runtime import get_optimal_device
from diffusers import StableDiffusionXLPipeline

from ..models import Text2ImageRequest, Text2ImageResponse


class LocalSDXLProvider(BaseProvider):
    """Local SDXL text-to-image provider using Diffusers."""

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = device if device else get_optimal_device()

        # Load the pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,  # type: ignore
            use_safetensors=True,
            variant="fp16" if self.device != "cpu" else None,
        )
        self.pipe.to(self.device)  # type: ignore

    def generate(self, request: Text2ImageRequest, output_path: str) -> Text2ImageResponse:
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self.device)  # type: ignore
            generator.manual_seed(request.seed)

        kwargs = {}
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt

        result = self.pipe(
            prompt=request.prompt,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            width=request.width,
            height=request.height,
            generator=generator,
            **kwargs,
        )

        if not result.images or len(result.images) == 0:  # type: ignore
            raise RuntimeError("Pipeline failed to generate an image.")

        output_image = result.images[0]  # type: ignore
        output_image.save(output_path)

        return Text2ImageResponse(
            output_path=output_path, metadata={"provider": "local_sdxl", "device": self.device}
        )
