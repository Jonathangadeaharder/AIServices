import time
from pathlib import Path

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger

from .models import Video2SubtitleRequest
from .providers import registry

app = typer.Typer(help="Video to subtitle transcription pipeline")
logger = get_logger(__name__)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    input_path: str = typer.Option(..., "--input", "-i", help="Path to input video file"),
    output: str = typer.Option(None, "--output", "-o", help="Path to save subtitle file (default: <input>.srt)"),
    language: str = typer.Option(None, "--language", "-l", help="Language code (e.g. 'en') or 'auto' for auto-detect"),
    burn_in: bool = typer.Option(False, "--burn-in", help="Burn subtitles into a new mp4 file"),
    verbose: bool = verbose_option,
    device: str = device_option,
):
    """Transcribe video to SRT/VTT subtitles."""
    if ctx.invoked_subcommand is not None:
        return

    input_path = str(Path(input_path).expanduser().resolve())
    if not Path(input_path).exists():
        typer.secho(f"Error: Input file not found: {input_path}", fg=typer.colors.RED, bold=True, err=True)
        raise typer.Exit(code=1)

    if output is None:
        output = str(Path(input_path).with_suffix(".srt"))

    request = Video2SubtitleRequest(
        video_path=input_path,
        language=language if language != "auto" else None,
        output_format="srt",
    )

    provider_name = "video2subtitle.mlx"
    logger.info(f"Using provider: {provider_name}")

    try:
        with create_progress_bar() as progress:
            task_id = progress.add_task("[cyan]Initializing provider...", total=None)
            provider = registry.get(provider_name, device=device)

            progress.update(task_id, description="[green]Extracting audio & transcribing...")
            start_time = time.time()
            response = provider.generate(request, output)
            elapsed = time.time() - start_time
            progress.update(task_id, description=f"[bold green]Done in {elapsed:.2f}s!")

        typer.echo(f"\nSubtitle file saved to: {response.output_path}")
        typer.echo(f"Entries: {len(response.entries)}")
        if response.language:
            typer.echo(f"Language: {response.language}")

        if burn_in:
            burned_output = str(Path(input_path).with_stem(Path(input_path).stem + "_subtitled"))
            progress_update = progress.add_task("[cyan]Burning subtitles into video...", total=None)
            _burn_subtitles(input_path, output, burned_output)
            typer.echo(f"\nBurned-in video saved to: {burned_output}")

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        typer.secho(f"\nError: {str(e)}", fg=typer.colors.RED, bold=True, err=True)
        raise typer.Exit(code=1) from e


def _burn_subtitles(video_path: str, subtitle_path: str, output_path: str) -> None:
    """Burn subtitles into video using ffmpeg."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. Install ffmpeg and ensure it is on PATH.")

    subtitle_path_escaped = subtitle_path.replace("\\", "/").replace(":", "\\:")
    filter_str = f"subtitles='{subtitle_path_escaped}'"

    cmd = [
        ffmpeg,
        "-i", video_path,
        "-vf", filter_str,
        "-c:a", "copy",
        "-y",
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg subtitle burn failed: {e.stderr.decode()}") from e
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg subtitle burn timed out after 600 seconds") from None


if __name__ == "__main__":
    app()
