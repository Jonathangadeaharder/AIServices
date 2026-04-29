from pathlib import Path

import mlx.core as mx
import soundfile as sf
from aiservices_core.providers import BaseProvider

from ..models import Text2SpeechRequest, Text2SpeechResponse


class FishMLXProvider(BaseProvider):
    """Text-to-speech provider using Fish S2 Pro MLX (Apple Silicon)."""

    def __init__(
        self,
        model_name: str = "fish-s2-pro",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            import mlx_speech

            self._model = mlx_speech.tts.load(self.model_name)
        return self._model

    def _build_text(self, request: Text2SpeechRequest) -> str:
        parts = []
        if request.emotion:
            parts.append(f"[{request.emotion}]")
        if request.tone:
            parts.append(f"[{request.tone}]")
        if request.effect:
            parts.append(f"[{request.effect}]")

        tags = "".join(parts)
        text = request.text.strip()
        return f"{tags} {text}" if tags else text

    def generate(
        self, request: Text2SpeechRequest, output_path: str
    ) -> Text2SpeechResponse:
        model = self._get_model()
        text = self._build_text(request)

        generate_kwargs = {"text": text}
        if request.reference_audio:
            generate_kwargs["reference_audio"] = request.reference_audio
        if request.reference_text:
            generate_kwargs["reference_text"] = request.reference_text

        result = model.generate(**generate_kwargs)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), result.waveform, result.sample_rate)

        return Text2SpeechResponse(
            output_path=str(out_path),
            metadata={
                "provider": "fish-s2-pro-mlx",
                "model": self.model_name,
                "sample_rate": result.sample_rate,
                "voice_id": request.voice_id,
            },
        )
