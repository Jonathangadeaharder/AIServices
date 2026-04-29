import urllib.error
import urllib.request
from pathlib import Path

from aiservices_core.providers import BaseProvider

from ..models import Text2SpeechRequest, Text2SpeechResponse


class FishSpeechProvider(BaseProvider):
    """Text-to-speech provider using Fish Speech HTTP API."""

    def __init__(
        self,
        api_url: str = "http://127.0.0.1:8090",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_url = api_url

    def _build_fish_text(self, request: Text2SpeechRequest) -> str:
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
        try:
            import ormsgpack
        except ImportError as e:
            raise ImportError("ormsgpack is required for Fish Speech API") from e

        if output_path is None:
            output_path = "output.wav"

        payload = {
            "text": self._build_fish_text(request),
            "references": [],
            "reference_id": request.voice_id or "default",
            "seed": 42,
            "temperature": 0.8,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            "chunk_length": 200,
            "max_new_tokens": 1024,
            "streaming": False,
            "format": "wav",
            "latency": "normal",
            "normalize": True,
            "use_memory_cache": "on",
        }

        try:
            data = ormsgpack.packb(payload)
            req = urllib.request.Request(
                f"{self.api_url}/v1/tts",
                data=data,
                headers={"Content-Type": "application/msgpack"},
            )

            with urllib.request.urlopen(req, timeout=300) as resp:
                audio_data = resp.read()

            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(audio_data)

            return Text2SpeechResponse(
                output_path=str(out_path),
                metadata={
                    "provider": "fish-speech-api",
                    "api_url": self.api_url,
                    "voice_id": request.voice_id,
                },
            )

        except urllib.error.URLError as e:
            raise RuntimeError(f"Fish Speech API failed: {e}") from e
