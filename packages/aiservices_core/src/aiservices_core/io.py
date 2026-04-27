import os
from pathlib import Path

from PIL import Image


def load_image(path_or_url: str | Path) -> Image.Image:
    """Load an image from a local path or URL."""
    path_str = str(path_or_url)
    if path_str.startswith("http://") or path_str.startswith("https://"):
        from io import BytesIO

        import requests

        response = requests.get(path_str)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    return Image.open(path_or_url).convert("RGB")


def save_image(img: Image.Image, path: str | Path):
    """Save an image to a path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img.save(path)
