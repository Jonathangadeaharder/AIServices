import shutil
import subprocess

from aiservices_core.errors import ProviderError
from aiservices_core.providers import BaseProvider

from ..models import Video2AudioRequest, Video2AudioResponse

FORMAT_CODEC_MAP = {
    "wav": "pcm_s16le",
    "mp3": "libmp3lame",
    "aac": "aac",
}


class FFmpegProvider(BaseProvider):
    """Audio extraction provider using FFmpeg."""

    def __init__(self, device: str = "auto", **kwargs):
        super().__init__(**kwargs)
        self.device = device

    def _find_ffmpeg(self) -> str:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise ProviderError("ffmpeg not found. Install FFmpeg and ensure it is on your PATH.")
        return ffmpeg_path

    def generate(self, request: Video2AudioRequest, output_path: str) -> Video2AudioResponse:
        ffmpeg_path = self._find_ffmpeg()

        codec = FORMAT_CODEC_MAP.get(request.output_format)
        if codec is None:
            raise ProviderError(
                f"Unsupported format: {request.output_format}. "
                f"Supported formats: {list(FORMAT_CODEC_MAP.keys())}"
            )

        cmd = [
            ffmpeg_path,
            "-i",
            request.video_path,
            "-vn",
            "-acodec",
            codec,
            "-ar",
            str(request.sample_rate),
        ]

        if request.mono:
            cmd.extend(["-ac", "1"])

        cmd.append(output_path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise ProviderError(f"FFmpeg failed: {e.stderr}") from e
        except FileNotFoundError as e:
            raise ProviderError(f"FFmpeg not found: {e}") from e

        return Video2AudioResponse(
            output_path=output_path,
            metadata={
                "provider": "ffmpeg",
                "format": request.output_format,
                "codec": codec,
                "sample_rate": request.sample_rate,
                "mono": request.mono,
                "ffmpeg_stderr": (
                    result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                ),
            },
        )
