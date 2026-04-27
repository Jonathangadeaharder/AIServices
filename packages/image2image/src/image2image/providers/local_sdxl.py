import torch
from aiservices_core.io import load_image
from aiservices_core.providers import BaseProvider
from aiservices_core.runtime import get_optimal_device
from diffusers import StableDiffusionXLImg2ImgPipeline

from ..models import Image2ImageRequest, Image2ImageResponse


class LocalSDXLProvider(BaseProvider):
    """Local SDXL image-to-image provider using Diffusers."""

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0", **kwargs):
        super().__init__(**kwargs)
        self.device = get_optimal_device()

        # Load the pipeline
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            use_safetensors=True,
            variant="fp16" if self.device != "cpu" else None,
        )
        self.pipe.to(self.device)

    def generate(self, request: Image2ImageRequest, output_path: str) -> Image2ImageResponse:
        init_image = load_image(request.image_path)

        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(request.seed)

        kwargs = {}
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt

        result = self.pipe(
            prompt=request.prompt,
            image=init_image,
            strength=request.strength,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            generator=generator,
            **kwargs,
        )

        output_image = result.images[0]
        output_image.save(output_path)

        return Image2ImageResponse(
            output_path=output_path, metadata={"provider": "local_sdxl", "device": self.device}
        )
