import time

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger

from .models import Speech2TextRequest
from .providers import registry

app = typer.Typer(help="Speech to text transcription pipeline")
logger = get_logger(__name__)


@app.command()
def transcribe(
    audio: str = typer.Option(..., "--audio", "-a", help="Path to input audio file"),
    output: str = typer.Option(None, "--output", "-o", help="Path to save transcription text"),
    model: str = typer.Option(
        "mlx-community/whisper-large-v3-mlx",
        "--model",
        help="Whisper model name",
    ),
    language: str = typer.Option(None, "--language", "-l", help="Language code (e.g. 'en')"),
    provider_name: str = typer.Option("speech2text.mlx", "--provider", help="Provider name"),
    verbose: bool = verbose_option,
    device: str = device_option,
):
    """Transcribe audio to text."""
    request = Speech2TextRequest(
        audio_path=audio,
        model_name=model,
        language=language,
    )

    logger.info(f"Using provider: {provider_name}")

    with create_progress_bar() as progress:
        task_id = progress.add_task("[cyan]Initializing provider...", total=None)
        provider = registry.get(provider_name, device=device)

        progress.update(task_id, description="[green]Transcribing audio...")
        start_time = time.time()
        response = provider.generate(request, output)
        elapsed = time.time() - start_time
        progress.update(task_id, description=f"[bold green]Done in {elapsed:.2f}s!")

    typer.echo(f"\nTranscription:\n{response.text}\n")
    if output:
        logger.info(f"Transcription saved to: {output}")


if __name__ == "__main__":
    app()
