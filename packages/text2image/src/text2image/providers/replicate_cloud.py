import os
from io import BytesIO

import replicate
import requests
from aiservices_core.errors import retry_api_call
from aiservices_core.providers import BaseProvider
from PIL import Image

from ..models import Text2ImageRequest, Text2ImageResponse


class ReplicateProvider(BaseProvider):
    """Cloud text-to-image provider using Replicate."""

    def __init__(
        self,
        model_id: str = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",  # noqa: E501
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_id = model_id

        if not os.environ.get("REPLICATE_API_TOKEN"):
            import logging

            logging.warning("REPLICATE_API_TOKEN not set. Replicate provider will fail if used.")

    @retry_api_call
    def _download_and_save(self, url: str, output_path: str):
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(output_path)

    def generate(self, request: Text2ImageRequest, output_path: str) -> Text2ImageResponse:
        input_data = {
            "prompt": request.prompt,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
            "width": request.width,
            "height": request.height,
        }

        if request.negative_prompt:
            input_data["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            input_data["seed"] = request.seed

        # Run the model
        output = replicate.run(self.model_id, input=input_data)

        # output is usually a list of URLs
        if isinstance(output, list) and len(output) > 0:
            result_url = output[0]
        else:
            result_url = str(output)

        self._download_and_save(result_url, output_path)

        return Text2ImageResponse(
            output_path=output_path,
            metadata={"provider": "replicate", "model_id": self.model_id, "url": result_url},
        )
