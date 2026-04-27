import json
import random
from pathlib import Path

from aiservices_core.comfyui import ComfyUIClient
from aiservices_core.providers import BaseProvider

from ..models import Text2ImageRequest, Text2ImageResponse


class ComfyUIProvider(BaseProvider):
    """Local text-to-image provider using ComfyUI."""

    def __init__(
        self,
        server_address: str = "127.0.0.1:8188",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.client = ComfyUIClient(server_address=server_address)

        # Load the default workflow
        workflow_path = Path(__file__).parent.parent / "workflows" / "flux2_t2i_api.json"
        with open(workflow_path) as f:
            self.workflow_template = json.load(f)

    def generate(self, request: Text2ImageRequest, output_path: str) -> Text2ImageResponse:
        prompt = json.loads(json.dumps(self.workflow_template))  # deep copy

        # Update prompt
        prompt["6"]["inputs"]["text"] = request.prompt

        # Update dimensions
        prompt["5"]["inputs"]["width"] = request.width
        prompt["5"]["inputs"]["height"] = request.height
        prompt["17"]["inputs"]["width"] = request.width
        prompt["17"]["inputs"]["height"] = request.height

        # Update steps
        prompt["17"]["inputs"]["steps"] = request.num_inference_steps

        # Update guidance
        prompt["26"]["inputs"]["guidance"] = request.guidance_scale

        # Update seed
        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        prompt["25"]["inputs"]["noise_seed"] = seed

        # Execute
        images = self.client.get_images(prompt)

        # Retrieve output from SaveImage node "9"
        if "9" not in images or not images["9"]:
            raise RuntimeError("ComfyUI workflow did not return an image.")

        image_data = images["9"][0]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(image_data)

        return Text2ImageResponse(
            output_path=output_path, metadata={"provider": "comfyui", "seed": seed}
        )
