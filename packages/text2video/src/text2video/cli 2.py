import time

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger

from .models import Text2VideoRequest
from .providers import registry

app = typer.Typer(help="Text to video generation pipeline")
logger = get_logger(__name__)


@app.command()
def generate(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Text prompt for video generation"),
    output: str = typer.Option(..., "--output", "-o", help="Path to save output"),
    provider_name: str = typer.Option(
        "text2video.mlx",
        "--provider",
        help="Provider name (text2video.mlx, text2video.comfyui)",
    ),
    negative_prompt: str = typer.Option(None, "--negative-prompt", "-n", help="Negative prompt"),
    width: int = typer.Option(640, "--width", help="Video width"),
    height: int = typer.Option(640, "--height", help="Video height"),
    num_frames: int = typer.Option(81, "--frames", help="Number of frames to generate"),
    steps: int = typer.Option(4, "--steps", help="Inference steps"),
    seed: int = typer.Option(None, "--seed", help="Random seed"),
    fps: int = typer.Option(16, "--fps", help="Frames per second"),
    verbose: bool = verbose_option,
    device: str = device_option,
):
    """Generate a video from text."""
    try:
        kwargs = {
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

        request = Text2VideoRequest(**kwargs)

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

    except Exception as e:
        logger.exception(f"Generation failed: {str(e)}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
