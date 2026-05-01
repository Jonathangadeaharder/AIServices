import time

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger
from aiservices_core.providers import registry

from .models import Text2SpeechRequest

app = typer.Typer(help="Text to speech generation pipeline")
logger = get_logger(__name__)


@app.command()
def generate(
    text: str = typer.Option(..., "--text", "-t", help="Text to convert to speech"),
    output: str = typer.Option(..., "--output", "-o", help="Path to save output audio file"),
    provider_name: str = typer.Option("text2speech.fish_mlx", "--provider", help="Provider name"),
    verbose: bool = verbose_option,
    device: str = device_option,
):
    """Generate speech from text."""
    logger.info(f"Using provider: {provider_name}")

    try:
        request = Text2SpeechRequest(
            text=text,
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
