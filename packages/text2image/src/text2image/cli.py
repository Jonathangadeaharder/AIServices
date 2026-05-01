import time

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger

from .models import Text2ImageRequest
from .providers import registry

app = typer.Typer(help="Text to image generation pipeline")
logger = get_logger(__name__)


@app.command()
def generate(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Text prompt for image generation"),
    output: str = typer.Option(..., "--output", "-o", help="Path to save output image"),
    provider_name: str = typer.Option(
        "text2image.mlx",
        "--provider",
        help="Provider name",
    ),
    negative_prompt: str | None = typer.Option(None, "--negative-prompt", help="Negative prompt"),
    guidance_scale: float = typer.Option(7.5, "--guidance-scale", help="Guidance scale"),
    steps: int = typer.Option(50, "--steps", help="Number of inference steps"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed"),
    width: int = typer.Option(1024, "--width", help="Width of the image"),
    height: int = typer.Option(1024, "--height", help="Height of the image"),
    verbose: bool = verbose_option,
    device: str = device_option,
):
    """Generate an image from text."""
    try:
        request = Text2ImageRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            seed=seed,
            width=width,
            height=height,
        )

        logger.info(f"Using provider: {provider_name}")

        with create_progress_bar() as progress:
            task_id = progress.add_task("[cyan]Initializing provider...", total=None)

            provider = registry.get(provider_name, device=device)

            progress.update(task_id, description="[green]Generating image...")
            start_time = time.time()

            response = provider.generate(request, output)

            elapsed = time.time() - start_time
            progress.update(task_id, description=f"[bold green]Done in {elapsed:.2f}s!")

        logger.info(f"Image saved to: {response.output_path}")
        logger.debug(f"Metadata: {response.metadata}")

    except Exception as e:
        logger.exception(f"Generation failed: {str(e)}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
