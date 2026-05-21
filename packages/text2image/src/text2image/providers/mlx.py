import random
import shutil
import subprocess
from pathlib import Path

from aiservices_core.providers import BaseProvider

from ..models import Text2ImageRequest, Text2ImageResponse


class MLXProvider(BaseProvider):
    DEFAULT_MODEL = "schnell"
    DEFAULT_STEPS = 4

    def __init__(
        self,
        model_name: str | None = None,
        quantize: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.quantize = quantize
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        executable = shutil.which("mflux-generate")
        if executable is None:
            raise ImportError("mflux-generate is not installed. Install text2image dependencies.")
        self._pipeline = executable

    def generate(
        self, request: Text2ImageRequest, output_path: str | None = None
    ) -> Text2ImageResponse:
        self._load_pipeline()
        if self._pipeline is None:
            raise RuntimeError("Pipeline failed to load")

        if output_path is None:
            output_path = "output.png"

        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._pipeline,
            "--model",
            self.model_name,
            "--prompt",
            request.prompt,
            "--output",
            output_path,
            "--width",
            str(request.width),
            "--height",
            str(request.height),
            "--steps",
            str(request.num_inference_steps or self.DEFAULT_STEPS),
            "--seed",
            str(seed),
            "--quantize",
            str(self.quantize),
        ]
        if request.guidance_scale is not None:
            cmd.extend(["--guidance", str(request.guidance_scale)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=180,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("mflux-generate timed out after 180 s") from exc
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or "mflux-generate failed")

        return Text2ImageResponse(
            output_path=output_path,
            metadata={
                "provider": "mlx",
                "model_name": self.model_name,
                "quantize": self.quantize,
                "seed": seed,
            },
        )
