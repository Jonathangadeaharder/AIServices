import time

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger

from .models import Text2SpeechRequest
from .providers import registry

app = typer.Typer(help="Text to speech generation pipeline")
logger = get_logger(__name__)


@app.command()
def generate(
    text: str = typer.Option(..., "--text", "-t", help="Text to convert to speech"),
    output: str = typer.Option(..., "--output", "-o", help="Path to save output audio file"),
    provider_name: str = typer.Option("text2speech.fish_mlx", "--provider", help="Provider name"),
    voice_id: str | None = typer.Option(
        None, "--voice-id", help="Voice ID for providers that support it"
    ),
    reference_audio: str | None = typer.Option(
        None, "--reference-audio", help="Reference audio path for voice cloning"
    ),
    reference_text: str | None = typer.Option(
        None, "--reference-text", help="Transcript of the reference audio"
    ),
    emotion: str | None = typer.Option(None, "--emotion", help="Emotion tag"),
    tone: str | None = typer.Option(None, "--tone", help="Tone tag"),
    effect: str | None = typer.Option(None, "--effect", help="Effect tag"),
    language: str | None = typer.Option(None, "--language", help="Language code"),
    temperature: float | None = typer.Option(
        None, "--temperature", help="Sampling temperature for neural TTS providers"
    ),
    verbose: bool = verbose_option,
    device: str = device_option,
):
    """Generate speech from text."""
    logger.info(f"Using provider: {provider_name}")

    try:
        request = Text2SpeechRequest(
            text=text,
            voice_id=voice_id,
            reference_audio=reference_audio,
            reference_text=reference_text,
            emotion=emotion,
            tone=tone,
            effect=effect,
            language=language,
            temperature=temperature,
        )

        with create_progress_bar() as progress:
            task_id = progress.add_task("[cyan]Initializing provider...", total=None)
            provider = registry.get(provider_name, device=device)

            progress.update(task_id, description="[green]Generating speech...")
            start_time = time.time()
            response = provider.generate(request, output)
            elapsed = time.time() - start_time
            progress.update(task_id, description=f"[bold green]Done in {elapsed:.2f}s!")

        logger.info(f"Speech saved to: {response.output_path}")
    except Exception as e:
        logger.error(f"Speech generation failed: {str(e)}")
        typer.echo(f"\n[bold red]Error:[/bold red] {str(e)}", err=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
