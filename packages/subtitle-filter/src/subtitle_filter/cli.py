import typer
from aiservices_core.logging import get_logger

from .srt_io import read_srt, write_srt

app = typer.Typer(help="Filter SRT subtitles by vocabulary level")
logger = get_logger(__name__)


@app.command()
def filter(
    input: str = typer.Option(None, "--input", "-i", help="Input SRT file (default: stdin)"),
    output: str = typer.Option(None, "--output", "-o", help="Output SRT file (default: stdout)"),
    vocab: str = typer.Option(..., "--vocab", "-v", help="Path to vocab CSV directory"),
    levels: str = typer.Option("A1,A2,B1", "--levels", "-l", help="Comma-separated vocab levels"),
    spacy_model: str = typer.Option("de_core_news_lg", "--spacy-model", help="spaCy model name"),
):
    """Filter SRT segments, keeping only those with words above the given vocabulary level."""
    import spacy

    from .filter import VocabFilter

    level_list = [lev.strip() for lev in levels.split(",")]
    subs = read_srt(input)
    logger.info(f"Filtering {len(subs)} segments (levels: {level_list})")

    vocab_set = VocabFilter.load_vocab(vocab, level_list)
    nlp = spacy.load(spacy_model, disable=["parser", "ner"])
    filtered = VocabFilter.filter_subs(subs, nlp, vocab_set)

    write_srt(filtered, output)
    logger.info(f"Done: {len(filtered)}/{len(subs)} segments kept")


if __name__ == "__main__":
    app()
