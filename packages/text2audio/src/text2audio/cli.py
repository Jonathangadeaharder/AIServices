import time

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger
from aiservices_core.providers import registry

from .models import Text2AudioRequest

app = typer.Typer(help="Text to audio generation pipeline")
logger = get_logger(__name__)


@app.callback(invoke_without_command=True)
def main(
    text: str = typer.Option(..., "--text", "-t", help="Text description of audio to generate"),
    output: str = typer.Option(..., "--output", "-o", help="Path to save output audio file"),
    voice: str = typer.Option("default", "--voice", help="Voice/speaker ID"),
    speed: float = typer.Option(1.0, "--speed", help="Playback speed multiplier"),
    seed: int | None = typer.Option(None, "--seed", "-s", help="Random seed"),
    verbose: bool = verbose_option,
    device: str = device_option,
):
    """Generate audio from a text description."""
    provider_name = "text2audio.fish_mlx"
    logger.info(f"Using provider: {provider_name}")

    try:
        request = Text2AudioRequest(
            text=text,
            voice=voice,
            speed=speed,
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
