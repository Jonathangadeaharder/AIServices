import json
from pathlib import Path

from .models import Episode


class EpisodeLoader:
    """Loads an Episode from a JSON file."""

    @staticmethod
    def load(path: str | Path) -> Episode:
        with open(path) as f:
            data = json.load(f)
        return Episode.model_validate(data)
