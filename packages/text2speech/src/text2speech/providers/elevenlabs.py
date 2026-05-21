import json
import os
import re
import tempfile
import urllib.error
import urllib.request
import wave
from pathlib import Path

from aiservices_core.providers import BaseProvider

from ..models import Text2SpeechRequest, Text2SpeechResponse


def _clean_text(text: str) -> str:
    text = re.sub(r"\[(?!pause).*?\]", "", text)
    text = text.replace("[pause:short]", " ")
    text = text.replace("[pause:medium]", " ... ")
    text = text.replace("[pause:long]", "\n\n")
    text = text.replace("[pause]", " ... ")
    return re.sub(r" +", " ", text).strip()


class ElevenLabsProvider(BaseProvider):
    """Text-to-speech provider using the ElevenLabs REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        model_id: str = "eleven_multilingual_v2",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVENLABS_TOKEN", "")
        self.voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID", "")
        self.model_id = model_id

    def generate(
        self, request: Text2SpeechRequest, output_path: str | None = None
    ) -> Text2SpeechResponse:
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is required for ElevenLabs TTS")

        voice_id = request.voice_id or self.voice_id
        if not voice_id:
            raise RuntimeError("ELEVENLABS_VOICE_ID or --voice-id is required")

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.close()
            output_path = tmp.name

        body: dict[str, str] = {
            "text": _clean_text(request.text),
            "model_id": self.model_id,
        }
        if request.language:
            body["language_code"] = request.language

        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            "?output_format=pcm_44100"
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "audio/pcm",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                pcm_data = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ElevenLabs API failed ({e.code}): {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"ElevenLabs API failed: {e}") from e

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(44100)
            wav_file.writeframes(pcm_data)

        return Text2SpeechResponse(
            output_path=str(out_path),
            metadata={
                "provider": "elevenlabs",
                "model_id": self.model_id,
                "voice_id": voice_id,
            },
        )
