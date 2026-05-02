import time
from typing import Literal, cast

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger

from .models import Audio2SubtitleRequest
from .providers import registry

app = typer.Typer(help="Audio to subtitle (SRT/VTT) transcription pipeline")
logger = get_logger(__name__)


@app.command()
def transcribe(
    audio: str = typer.Option(..., "--audio", "-a", help="Path to input audio file"),
    output: str = typer.Option(..., "--output", "-o", help="Path to output subtitle file"),
    format: str = typer.Option("srt", "--format", "-f", help="Output format: srt or vtt"),
    language: str = typer.Option(None, "--language", "-l", help="Language code (e.g. 'de')"),
    model: str = typer.Option(
        "mlx-community/whisper-large-v3-turbo",
        "--model",
        help="Whisper model name",
    ),
    no_word_timestamps: bool = typer.Option(
        False, "--no-word-timestamps", help="Disable word-level timestamps"
    ),
    provider_name: str = typer.Option("audio2subtitle.mlx", "--provider", help="Provider name"),
    verbose: bool = verbose_option,
    device: str = device_option,
):
    """Transcribe audio to subtitle file (SRT/VTT)."""
    request = Audio2SubtitleRequest(
        audio_path=audio,
        language=language,
        output_format=cast(Literal["srt", "vtt"], format),
        model_name=model,
        word_timestamps=not no_word_timestamps,
    )

    logger.info(f"Using provider: {provider_name}")

    try:
        with create_progress_bar() as progress:
            task_id = progress.add_task("[cyan]Initializing provider...", total=None)
            provider = registry.get(provider_name, device=device)

            progress.update(task_id, description="[green]Transcribing audio to subtitles...")
            start_time = time.time()
            response = provider.generate(request, output)
            elapsed = time.time() - start_time
            progress.update(task_id, description=f"[bold green]Done in {elapsed:.2f}s!")

        typer.echo(f"\nSubtitle file saved to: {response.output_path}")
        typer.echo(f"Detected language: {response.language}")
        typer.echo(f"Entries: {len(response.entries)}")
    except Exception as e:
        logger.error(f"Subtitle generation failed: {str(e)}")
        typer.secho(f"\nError: {str(e)}", fg=typer.colors.RED, bold=True, err=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
