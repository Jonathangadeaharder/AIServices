import time

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger

from .models import Image2VideoRequest
from .providers import registry

app = typer.Typer(help="Image to video generation pipeline")
logger = get_logger(__name__)


@app.command()
def generate(
    input: str = typer.Option(..., "--input", "-i", help="Path to the input image"),
    output: str = typer.Option(..., "--output", "-o", help="Path to save output video"),
    seconds: int = typer.Option(4, "--seconds", "-s", help="Video duration in seconds"),
    fps: int = typer.Option(24, "--fps", help="Frames per second"),
    prompt: str = typer.Option("", "--prompt", "-p", help="Text prompt for video generation"),
    provider_name: str = typer.Option(
        "image2video.mlx",
        "--provider",
        help="Provider name (image2video.mlx)",
    ),
    negative_prompt: str = typer.Option(None, "--negative-prompt", "-n", help="Negative prompt"),
    width: int = typer.Option(640, "--width", help="Video width"),
    height: int = typer.Option(640, "--height", help="Video height"),
    steps: int = typer.Option(4, "--steps", help="Inference steps"),
    seed: int = typer.Option(None, "--seed", help="Random seed"),
    verbose: bool = verbose_option,
    device: str = device_option,
):
    """Generate a video from an image."""
    num_frames = seconds * fps

    kwargs = {
        "image_path": input,
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "seed": seed,
        "fps": fps,
    }
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt

    request = Image2VideoRequest(**kwargs)

    logger.info(f"Using provider: {provider_name}")

    with create_progress_bar() as progress:
        task_id = progress.add_task("[cyan]Initializing provider...", total=None)

        provider = registry.get(provider_name, device=device)

        progress.update(task_id, description="[green]Generating video...")
        start_time = time.time()

        response = provider.generate(request, output)

        elapsed = time.time() - start_time
        progress.update(
            task_id,
            description=f"[bold green]Done in {elapsed:.2f}s!",
        )

    logger.info(f"Video saved to: {response.output_path}")
    logger.debug(f"Metadata: {response.metadata}")


if __name__ == "__main__":
    app()
