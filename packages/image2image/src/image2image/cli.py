import time

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger

from .models import Image2ImageRequest
from .providers import registry

app = typer.Typer(help="Image to image generation pipeline")
logger = get_logger(__name__)


@app.command()
def main(
    input_path: str = typer.Option(..., "--input", "-i", help="Path to input"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Text prompt"),
    output_path: str = typer.Option(..., "--output", "-o", help="Path to save output"),
    provider_name: str = typer.Option(
        "image2image.clothing",
        "--provider",
        help="Provider name",
    ),
    strength: float = typer.Option(0.5, "--strength", "-s", help="Transformation strength"),
    guidance_scale: float = typer.Option(7.5, "--guidance", help="Guidance scale"),
    steps: int = typer.Option(50, "--steps", help="Number of inference steps"),
    seed: int = typer.Option(None, "--seed", help="Random seed"),
    negative_prompt: str = typer.Option(
        None, "--negative-prompt", "-n", help="Negative text prompt"
    ),  # noqa: E501
    verbose: bool = verbose_option,
    device: str = device_option,
):
    try:
        request = Image2ImageRequest(
            image_path=input_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            seed=seed,
        )

        logger.info(f"Using provider: {provider_name}")

        with create_progress_bar() as progress:
            task_id = progress.add_task("[cyan]Initializing provider...", total=None)

            provider = registry.get(provider_name, device=device)

            progress.update(task_id, description="[green]Generating image...")
            start_time = time.time()

            response = provider.generate(request, output_path)

            elapsed = time.time() - start_time
            progress.update(task_id, description=f"[bold green]Done in {elapsed:.2f}s!")

        logger.info(f"Image saved to: {response.output_path}")
        logger.debug(f"Metadata: {response.metadata}")

    except Exception as e:
        logger.exception(f"Generation failed: {str(e)}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
