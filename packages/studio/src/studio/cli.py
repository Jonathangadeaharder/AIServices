import typer
from pathlib import Path
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger

from .loader import EpisodeLoader
from .orchestrator.showrunner import Showrunner

app = typer.Typer(help="AI Showrunner Studio Orchestrator")
logger = get_logger(__name__)


@app.command()
def render(
    script: Path = typer.Option(..., "--script", "-s", help="Path to episode JSON script"),
    output_dir: Path = typer.Option("output/rendered", "--output", "-o", help="Output directory"),
    tts_provider: str = typer.Option("text2speech.fish", "--tts-provider"),
    t2v_provider: str = typer.Option("text2video.comfyui", "--t2v-provider"),
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
        )

        with create_progress_bar() as progress:
            task_id = progress.add_task(f"[cyan]Rendering {episode.title}...", total=len(episode.scenes))
            
            for scene in episode.scenes:
                progress.update(task_id, description=f"[green]Rendering Scene {scene.scene_id}...")
                showrunner.render_scene(scene, episode.cast)
                progress.advance(task_id)

            progress.update(task_id, description="[bold green]Finalizing episode...")
            final_path = showrunner.render_episode(episode)
            
        typer.echo(f"\n[bold green]✓ Episode rendered successfully![/bold green]")
        typer.echo(f"Output path: {final_path}")

    except Exception as e:
        logger.exception(f"Render failed: {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
