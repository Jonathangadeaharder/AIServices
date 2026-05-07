import time
from typing import Literal, cast

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger

from .models import Video2AudioRequest
from .providers import registry

app = typer.Typer(help="Extract audio from video files")
logger = get_logger(__name__)


@app.command()
def extract(
    input_path: str = typer.Option(..., "--input", "-i", help="Path to input video file"),
    output: str = typer.Option(..., "--output", "-o", help="Path to save output audio"),
    codec: str = typer.Option("wav", "--codec", "-c", help="Output audio format (wav, mp3, aac)"),
    provider_name: str = typer.Option(
        "video2audio.ffmpeg",
        "--provider",
        help="Provider name",
    ),
    verbose: bool = verbose_option,
    device: str = device_option,
):
    """Extract audio from a video file."""
    logger.info(f"Using provider: {provider_name}")

    try:
        request = Video2AudioRequest(
            video_path=input_path,
            output_format=cast(Literal["wav", "mp3", "aac"], codec),
        )

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
    except Exception as e:
        logger.error(f"Audio extraction failed: {str(e)}")
        typer.echo(f"\n[bold red]Error:[/bold red] {str(e)}", err=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
