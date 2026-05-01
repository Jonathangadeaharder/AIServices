import time
from typing import Literal, cast

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger

from .models import Audio2SubtitleRequest
from .providers import registry

app = typer.Typer(help="Audio to subtitle (SRT/VTT) transcription pipeline")
logger = get_logger(__name__)


@app.command()
def transcribe(
    audio: str = typer.Option(..., "--audio", "-a", help="Path to input audio file"),
    output: str = typer.Option(..., "--output", "-o", help="Path to output subtitle file"),
    format: str = typer.Option("srt", "--format", "-f", help="Output format: srt or vtt"),
    language: str = typer.Option(None, "--language", "-l", help="Language code (e.g. 'de')"),
    model: str = typer.Option(
        "mlx-community/whisper-large-v3-turbo",
        "--model",
        help="Whisper model name",
    ),
    no_word_timestamps: bool = typer.Option(
        False, "--no-word-timestamps", help="Disable word-level timestamps"
    ),
    translate_to: str = typer.Option(
        None, "--translate-to", help="Target language for translation (e.g. 'es')"
    ),
    translation_model: str = typer.Option(
        "Helsinki-NLP/opus-mt-tc-big-de-es",
        "--translation-model",
        help="Translation model name (for tokenizer)",
    ),
    ct2_model_path: str = typer.Option(
        "", "--ct2-model-path", help="Path to CTranslate2-converted model directory"
    ),
    vocab_filter: str = typer.Option(
        None, "--vocab-filter", help="Path to vocab CSV directory for filtering"
    ),
    vocab_levels: str = typer.Option(
        "A1,A2,B1", "--vocab-levels", help="Comma-separated vocab levels"
    ),
    provider_name: str = typer.Option("audio2subtitle.mlx", "--provider", help="Provider name"),
    verbose: bool = verbose_option,
    device: str = device_option,
):
    """Transcribe audio to subtitle file (SRT/VTT), with optional filtering and translation."""
    levels = [l.strip() for l in vocab_levels.split(",")]

    request = Audio2SubtitleRequest(
        audio_path=audio,
        language=language,
        output_format=cast(Literal["srt", "vtt"], format),
        model_name=model,
        word_timestamps=not no_word_timestamps,
        translate_to=translate_to,
        translation_model=translation_model,
        ct2_model_path=ct2_model_path,
        vocab_filter_path=vocab_filter,
        vocab_levels=levels,
    )

    logger.info(f"Using provider: {provider_name}")

    try:
        with create_progress_bar() as progress:
            task_id = progress.add_task("[cyan]Initializing provider...", total=None)
            provider = registry.get(provider_name, device=device)

            progress.update(task_id, description="[green]Transcribing audio to subtitles...")
            start_time = time.time()
            response = provider.generate(request, output)
            elapsed = time.time() - start_time
            progress.update(task_id, description=f"[bold green]Done in {elapsed:.2f}s!")

        typer.echo(f"\nSubtitle file saved to: {response.output_path}")
        typer.echo(f"Detected language: {response.language}")
        typer.echo(f"Entries: {len(response.entries)}")
    except Exception as e:
        logger.error(f"Subtitle generation failed: {str(e)}")
        typer.secho(f"\nError: {str(e)}", fg=typer.colors.RED, bold=True, err=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
