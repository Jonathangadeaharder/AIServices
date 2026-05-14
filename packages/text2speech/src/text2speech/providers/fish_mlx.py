import tempfile
from pathlib import Path

from aiservices_core.providers import BaseProvider

from ..models import Text2SpeechRequest, Text2SpeechResponse


class FishMLXProvider(BaseProvider):
    """Text-to-speech provider using Fish S2 Pro via mlx-audio (Apple Silicon)."""

    def __init__(
        self,
        model_name: str = "mlx-community/fish-audio-s2-pro-bf16",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from mlx_audio.tts.utils import load_model  # type: ignore[import-not-found]

            model_path: Path | str = (
                Path(self.model_name) if Path(self.model_name).exists() else self.model_name
            )
            self._model = load_model(model_path)
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
        self, request: Text2SpeechRequest, output_path: str | None = None
    ) -> Text2SpeechResponse:
        from mlx_audio.tts.generate import generate_audio  # type: ignore[import-not-found]

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.close()
            output_path = tmp.name

        model = self._get_model()
        text = self._build_text(request)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        prefix = out_path.stem

        generate_kwargs: dict = {
            "model": model,
            "text": text,
            "file_prefix": str(out_path.parent / prefix),
        }
        if request.reference_audio:
            generate_kwargs["ref_audio"] = request.reference_audio

        generate_audio(**generate_kwargs)

        return Text2SpeechResponse(
            output_path=str(out_path),
            metadata={
                "provider": "fish-s2-pro-mlx",
                "model": self.model_name,
                "voice_id": request.voice_id,
            },
        )
