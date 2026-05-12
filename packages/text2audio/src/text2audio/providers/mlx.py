import random
import wave
from pathlib import Path

from aiservices_core.providers import BaseProvider

from ..models import Text2AudioRequest, Text2AudioResponse


class MLXProvider(BaseProvider):
    """Local text-to-audio provider using MLX (Apple Silicon).

    Placeholder provider that generates a silent WAV file.
    Full MusicGen/AudioLDM2 pipeline to be implemented.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(
        self, request: Text2AudioRequest, output_path: str | None = None
    ) -> Text2AudioResponse:
        if request.output_format != "wav":
            raise ValueError("MLXProvider currently supports only output_format='wav'")

        if output_path is None:
            output_path = f"output.{request.output_format}"

        effective_seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        sample_rate = 44100
        num_samples = int(request.duration_seconds * sample_rate)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with wave.open(output_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"\x00\x00" * num_samples)

        return Text2AudioResponse(
            output_path=output_path,
            duration_seconds=request.duration_seconds,
            metadata={
                "provider": "mlx",
                "seed": effective_seed,
                "note": "Placeholder - silent WAV. Full MusicGen pipeline not yet implemented",
            },
        )
