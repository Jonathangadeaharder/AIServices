import typer
from aiservices_core.logging import get_logger

from .srt_io import read_srt, write_srt

app = typer.Typer(help="Translate SRT subtitles between languages")
logger = get_logger(__name__)


@app.command()
def translate(
    input: str = typer.Option(None, "--input", "-i", help="Input SRT file (default: stdin)"),
    output: str = typer.Option(None, "--output", "-o", help="Output SRT file (default: stdout)"),
    to: str = typer.Option(..., "--to", "-t", help="Target language code (e.g. 'es')"),
    model: str = typer.Option(
        "Helsinki-NLP/opus-mt-tc-big-de-es", "--model", "-m", help="HuggingFace model name (for tokenizer)"
    ),
    ct2_model: str = typer.Option(..., "--ct2-model", "-c", help="Path to CTranslate2 model directory"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Translation batch size"),
):
    """Translate an SRT file from one language to another."""
    from .translator import MarianTranslator

    subs = read_srt(input)
    texts = [sub.text for sub in subs]
    logger.info(f"Translating {len(texts)} segments → {to}")

    translator = MarianTranslator(ct2_model, model)
    translated = translator.translate_batch(texts, batch_size)

    for sub, trans in zip(subs, translated):
        sub.text = trans

    write_srt(subs, output)
    logger.info(f"Done: {len(subs)} segments translated")


if __name__ == "__main__":
    app()
