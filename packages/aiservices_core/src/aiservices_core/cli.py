import typer

from .logging import setup_logging


def verbose_callback(verbose: bool):
    """Callback to setup rich logging based on verbosity."""
    setup_logging(debug=verbose)


verbose_option = typer.Option(
    False, "--verbose", "-v", help="Enable verbose output", callback=verbose_callback, is_eager=True
)

device_option = typer.Option("auto", "--device", "-d", help="Compute device (auto, cpu, cuda, mps)")
