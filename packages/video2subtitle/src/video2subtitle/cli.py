import time

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger

from .models import Video2SubtitleRequest
from .providers import registry

app = typer.Typer(help="Video to subtitle transcription pipeline")
logger = get_logger(__name__)


@app.command()
def transcribe(
    video: str = typer.Option(..., "--video", help="Path to input video file"),
    output: str = typer.Option(..., "--output", "-o", help="Path to save subtitle file"),
    format: str = typer.Option("srt", "--format", "-f", help="Output format: srt or vtt"),
    language: str = typer.Option(None, "--language", "-l", help="Language code (e.g. 'en')"),
    model: str = typer.Option(
        "mlx-community/whisper-large-v3-turbo",
        "--model",
        help="Whisper model name",
    ),
    no_word_timestamps: bool = typer.Option(
        False, "--no-word-timestamps", help="Disable word-level timestamps"
    ),
    provider_name: str = typer.Option("video2subtitle.mlx", "--provider", help="Provider name"),
    verbose: bool = verbose_option,
    device: str = device_option,
):
    """Transcribe video to SRT/VTT subtitles."""
    request = Video2SubtitleRequest(
        video_path=video,
        language=language,
        output_format=format,
        model_name=model,
        word_timestamps=not no_word_timestamps,
    )

    logger.info(f"Using provider: {provider_name}")

    try:
        with create_progress_bar() as progress:
            task_id = progress.add_task("[cyan]Initializing provider...", total=None)
            provider = registry.get(provider_name, device=device)

            progress.update(task_id, description="[green]Extracting audio & transcribing...")
            start_time = time.time()
            response = provider.generate(request, output)
            elapsed = time.time() - start_time
            progress.update(task_id, description=f"[bold green]Done in {elapsed:.2f}s!")

        typer.echo(f"\nSubtitle file saved to: {response.output_path}")
        typer.echo(f"Entries: {len(response.entries)}")
        if response.language:
            typer.echo(f"Language: {response.language}")
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        typer.secho(f"\nError: {str(e)}", fg=typer.colors.RED, bold=True, err=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
