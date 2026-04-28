from pathlib import Path

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger

from .loader import EpisodeLoader
from .orchestrator.showrunner import Showrunner

app = typer.Typer(help="AI Showrunner Studio Orchestrator")
logger = get_logger(__name__)


@app.command()
def render(  # noqa: B008
    script: Path = typer.Option(  # noqa: B008
        ..., "--script", "-s", help="Path to episode JSON script"
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        "output/rendered", "--output", "-o", help="Output directory"
    ),
    tts_provider: str = typer.Option(  # noqa: B008
        "text2speech.fish", "--tts-provider"
    ),
    t2v_provider: str = typer.Option(  # noqa: B008
        "text2video.comfyui", "--t2v-provider"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate rendering without calling providers"),
    keep_temp: bool = typer.Option(False, "--keep-temp", help="Keep temporary files after rendering"),
    verbose: bool = verbose_option,
    device: str = device_option,
):
    """Render a full episode from a script."""
    try:
        episode = EpisodeLoader.load(script)
        logger.info(f"Loaded script: {episode.title}")

        showrunner = Showrunner(
            output_dir=output_dir,
            tts_provider=tts_provider,
            t2v_provider=t2v_provider,
            device=device,
            dry_run=dry_run,
        )

        with create_progress_bar() as progress:
            task_id = progress.add_task(
                f"[cyan]Rendering {episode.title}...", total=100
            )
            
            # Since render_episode is a blocking call currently, we just update to 'rendering'
            progress.update(task_id, description=f"[green]Rendering {episode.title}...")
            final_path = showrunner.render_episode(episode)
            
            if not keep_temp:
                showrunner.cleanup()
                
            progress.update(task_id, completed=100)

        typer.echo("\n[bold green]✓ Episode rendered successfully![/bold green]")
        typer.echo(f"Output path: {final_path}")

    except Exception as e:
        logger.exception(f"Render failed: {str(e)}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
