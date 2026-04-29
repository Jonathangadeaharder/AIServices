import time

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger

from .models import AudioCategory, Text2AudioRequest
from .providers import registry

app = typer.Typer(help="Text to audio (music, SFX, ambient) generation pipeline")
logger = get_logger(__name__)


@app.command()
def generate(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Text prompt describing the audio"),
    output: str = typer.Option(..., "--output", "-o", help="Path to save output audio file"),
    category: str = typer.Option(
        "music", "--category", "-c", help="Audio category (music, sfx, ambient, speech)"
    ),
    duration: float = typer.Option(10.0, "--duration", help="Duration in seconds"),
    format: str = typer.Option("wav", "--format", "-f", help="Output format (wav, mp3)"),
    seed: int | None = typer.Option(None, "--seed", "-s", help="Random seed"),
    provider_name: str = typer.Option("text2audio.replicate", "--provider", help="Provider name"),
    verbose: bool = verbose_option,
    device: str = device_option,
):
    """Generate audio from a text prompt."""
    logger.info(f"Using provider: {provider_name}")

    try:
        request = Text2AudioRequest(
            prompt=prompt,
            duration_seconds=duration,
            output_format=format,
            category=AudioCategory(category),
            seed=seed,
        )

        with create_progress_bar() as progress:
            task_id = progress.add_task("[cyan]Initializing provider...", total=None)
            provider = registry.get(provider_name, device=device)

            progress.update(task_id, description="[green]Generating audio...")
            start_time = time.time()
            response = provider.generate(request, output)
            elapsed = time.time() - start_time
            progress.update(task_id, description=f"[bold green]Done in {elapsed:.2f}s!")

        logger.info(f"Audio saved to: {response.output_path}")
    except Exception as e:
        logger.error(f"Audio generation failed: {str(e)}")
        typer.echo(f"\n[bold red]Error:[/bold red] {str(e)}", err=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
