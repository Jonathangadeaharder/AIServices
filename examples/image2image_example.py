"""Image-to-image example using the Python API.

Transforms an existing image with a text prompt using Flux 2 MLX on Apple Silicon.

Usage:
    uv run python examples/image2image_example.py

Requires an input image at input.jpg (or change INPUT_PATH below).
"""

from pathlib import Path

from image2image.client import generate

INPUT_PATH = Path("input.jpg")
OUTPUT_PATH = Path("output_image2image.jpg")


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"Create or place an image at {INPUT_PATH} to run this example.")
        return

    output = generate(
        image_path=INPUT_PATH,
        prompt="watercolor painting of a landscape",
        output_path=OUTPUT_PATH,
        strength=0.6,
        guidance_scale=7.5,
        num_inference_steps=30,
        seed=42,
    )
    print(f"Generated image saved to {output}")


if __name__ == "__main__":
    main()
