import mlx_whisper
from aiservices_core.providers import BaseProvider

from ..models import Speech2TextRequest, Speech2TextResponse


class MLXWhisperProvider(BaseProvider):
    """Speech-to-text provider using mlx-whisper (Apple Silicon)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(
        self,
        request: Speech2TextRequest,
        output_path: str | None = None,
    ) -> Speech2TextResponse:
        result = mlx_whisper.transcribe(
            request.audio_path,
            path_or_hf_repo=request.model_name,
            language=request.language,
        )

        text = result.get("text", "").strip()

        if output_path:
            with open(output_path, "w") as f:
                f.write(text)

        return Speech2TextResponse(
            text=text,
            metadata={
                "provider": "mlx-whisper",
                "model": request.model_name,
                "language": result.get("language"),
                "segments": result.get("segments"),
            },
        )
