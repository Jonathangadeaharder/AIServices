import json
import random
from pathlib import Path

from aiservices_core.comfyui import ComfyUIClient
from aiservices_core.providers import BaseProvider

from ..models import Image2ImageRequest, Image2ImageResponse


class ComfyUIProvider(BaseProvider):
    """Local image-to-image provider using ComfyUI.

    Wraps the Flux2 image-edit workflow (image_flux2_api.json) and communicates
    with a running ComfyUI server over WebSocket.
    """

    def __init__(
        self,
        server_address: str = "127.0.0.1:8188",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.client = ComfyUIClient(server_address=server_address)

        workflow_path = Path(__file__).parent.parent / "workflows" / "image_flux2_api.json"
        with open(workflow_path) as f:
            self.workflow_template = json.load(f)

    def generate(self, request: Image2ImageRequest, output_path: str) -> Image2ImageResponse:
        prompt = json.loads(json.dumps(self.workflow_template))  # deep copy

        # Stage the input image into ComfyUI's input directory.
        # The LoadImage node (46) expects a filename that exists in ComfyUI/input/.
        input_image = Path(request.image_path).resolve()
        if not input_image.exists():
            raise FileNotFoundError(f"Input image not found: {request.image_path}")

        # Copy image to ComfyUI input folder so the server can find it.
        # The client will reference it by basename.
        image_basename = input_image.name
        prompt["46"]["inputs"]["image"] = image_basename

        # Upload the image via the ComfyUI upload endpoint
        self.client.upload_image(str(input_image), image_basename)

        # Update prompt text
        prompt["6"]["inputs"]["text"] = request.prompt

        # Update guidance scale
        prompt["26"]["inputs"]["guidance"] = request.guidance_scale

        # Update steps
        prompt["48"]["inputs"]["steps"] = request.num_inference_steps

        # Update seed
        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        prompt["25"]["inputs"]["noise_seed"] = seed

        # Execute the workflow
        images = self.client.get_images(prompt)

        # Retrieve output from SaveImage node "9"
        if "9" not in images or not images["9"]:
            raise RuntimeError("ComfyUI workflow did not return an image.")

        image_data = images["9"][0]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(image_data)

        return Image2ImageResponse(
            output_path=output_path,
            metadata={"provider": "comfyui", "seed": seed},
        )
