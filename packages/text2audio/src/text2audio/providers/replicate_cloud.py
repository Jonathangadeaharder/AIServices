import os
import urllib.request
from typing import Any

from aiservices_core.errors import ProviderError, retry_api_call
from aiservices_core.providers import BaseProvider

from ..models import AudioCategory, Text2AudioRequest, Text2AudioResponse

MODEL_MAP: dict[str, str] = {
    "meta/musicgen": (
        "meta/musicgen:67e5f731cf5f398e2b48fcf9c6a05e14e0c25b0e15e2b7f1e1e0e0e0e0e0e0e0"
    ),
}

CATEGORY_PARAMS: dict[AudioCategory, dict[str, Any]] = {
    AudioCategory.MUSIC: {"model": "melody"},
    AudioCategory.SFX: {"model": "melody"},
    AudioCategory.AMBIENT: {"model": "melody"},
    AudioCategory.SPEECH: {"model": "melody"},
}


class ReplicateProvider(BaseProvider):
    """Cloud text-to-audio provider using Replicate."""

    def __init__(self, device: str = "auto", **kwargs):
        super().__init__(**kwargs)
        self.device = device

        if not os.environ.get("REPLICATE_API_TOKEN"):
            import logging

            logging.warning("REPLICATE_API_TOKEN not set. Replicate provider will fail if used.")

    @retry_api_call
    def _download_audio(self, url: str, output_path: str):
        urllib.request.urlretrieve(url, output_path)

    def generate(self, request: Text2AudioRequest, output_path: str) -> Text2AudioResponse:
        input_data: dict[str, Any] = {
            "prompt": request.prompt,
            "duration": request.duration_seconds,
            "output_format": request.output_format,
        }

        input_data.update(CATEGORY_PARAMS.get(request.category, {}))

        if request.negative_prompt:
            input_data["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            input_data["seed"] = request.seed

        model_id = MODEL_MAP.get(request.model_version, request.model_version)

        try:
            import replicate

            output = replicate.run(model_id, input=input_data)
        except Exception as e:
            raise ProviderError(f"Replicate audio generation failed for {model_id}") from e

        result_url = self._extract_url(output)

        try:
            self._download_audio(result_url, output_path)
        except Exception as e:
            raise ProviderError(f"Failed to download audio from {result_url}") from e

        return Text2AudioResponse(
            output_path=output_path,
            duration_seconds=request.duration_seconds,
            metadata={
                "provider": "replicate",
                "model_id": model_id,
                "category": request.category.value,
                "url": result_url,
            },
        )

    def _extract_url(self, output: Any) -> str:
        result_url = None
        if isinstance(output, list) and len(output) > 0:
            result_url = str(output[0])
        elif isinstance(output, dict):
            for key in ["url", "audio", "output"]:
                if key in output:
                    result_url = str(output[key])
                    break
        elif isinstance(output, str):
            result_url = output

        if not result_url:
            raise ProviderError(f"Failed to parse output URL from Replicate response: {output}")

        return result_url
