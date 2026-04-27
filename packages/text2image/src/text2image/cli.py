import time

import typer
from aiservices_core.cli import create_progress_bar, get_app
from aiservices_core.errors import handle_cli_exceptions
from aiservices_core.providers import registry

from .models import Text2ImageRequest
from .providers import __name__  # Ensures providers are registered

app = get_app()


@app.command()
@handle_cli_exceptions
def generate(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Text prompt for image generation"),
    output: str = typer.Option(..., "--output", "-o", help="Path to save output image"),
    provider_name: str = typer.Option(
        "local_sdxl", "--provider", help="Provider to use (local_sdxl, replicate)"
    ),
    negative_prompt: str | None = typer.Option(None, "--negative-prompt", help="Negative prompt"),
    guidance_scale: float = typer.Option(7.5, "--guidance-scale", help="Guidance scale"),
    steps: int = typer.Option(50, "--steps", help="Number of inference steps"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed"),
    width: int = typer.Option(1024, "--width", help="Width of the image"),
    height: int = typer.Option(1024, "--height", help="Height of the image"),
    device: str | None = typer.Option(
        None, "--device", help="Force device for local provider (e.g., mps, cuda, cpu)"
    ),
):
    """Generate an image from text."""
    typer.echo(f"Starting Text-to-Image generation using {provider_name}")

    request = Text2ImageRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        seed=seed,
        width=width,
        height=height,
    )

    try:
        with create_progress_bar() as progress:
            task_id = progress.add_task("[cyan]Initializing provider...", total=None)

            provider = registry.get(provider_name, device=device)

            progress.update(task_id, description="[green]Generating image...")
            start_time = time.time()

            response = provider.generate(request, output)

            elapsed = time.time() - start_time
            progress.update(task_id, description="[bold green]Done!")

        typer.echo(f"Success! Image saved to {response.output_path} in {elapsed:.2f}s")
        typer.echo(f"Metadata: {response.metadata}")

    except Exception as e:
        typer.secho(f"Error generating image: {str(e)}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
