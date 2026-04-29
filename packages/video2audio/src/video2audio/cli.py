import time

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger

from .models import Video2AudioRequest
from .providers import registry

app = typer.Typer(help="Video to audio extraction and generation pipeline")
logger = get_logger(__name__)


@app.command()
def extract(
    video: str = typer.Option(..., "--video", "-i", help="Path to input video file"),
    output: str = typer.Option(..., "--output", "-o", help="Path to save output audio"),
    output_format: str = typer.Option(
        "wav", "--format", "-f", help="Output audio format (wav, mp3, aac)"
    ),
    sample_rate: int = typer.Option(44100, "--sample-rate", "-r", help="Output sample rate in Hz"),
    mono: bool = typer.Option(True, "--mono/--stereo", help="Convert to mono"),
    provider_name: str = typer.Option(
        "video2audio.ffmpeg",
        "--provider",
        help="Provider name",
    ),
    verbose: bool = verbose_option,
    device: str = device_option,
):
    """Extract audio from a video file."""
    request = Video2AudioRequest(
        video_path=video,
        output_format=output_format,
        sample_rate=sample_rate,
        mono=mono,
    )

    logger.info(f"Using provider: {provider_name}")

    with create_progress_bar() as progress:
        task_id = progress.add_task("[cyan]Initializing provider...", total=None)

        provider = registry.get(provider_name, device=device)

        progress.update(task_id, description="[green]Extracting audio...")
        start_time = time.time()

        response = provider.generate(request, output)

        elapsed = time.time() - start_time
        progress.update(
            task_id,
            description=f"[bold green]Done in {elapsed:.2f}s!",
        )

    logger.info(f"Audio saved to: {response.output_path}")
    if response.duration_seconds is not None:
        logger.info(f"Duration: {response.duration_seconds:.2f}s")
    logger.debug(f"Metadata: {response.metadata}")


if __name__ == "__main__":
    app()
