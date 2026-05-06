"""Text-to-image example using the Python API.

Generates an image from a text prompt using Flux 2 MLX on Apple Silicon.

Usage:
    uv run python examples/text2image_example.py
"""

from pathlib import Path

from text2image.client import generate

OUTPUT_PATH = Path("output_text2image.png")


def main() -> None:
    output = generate(
        prompt="a futuristic cityscape at sunset, cyberpunk style, highly detailed",
        output_path=OUTPUT_PATH,
        width=1024,
        height=1024,
        guidance_scale=7.5,
        num_inference_steps=30,
        seed=42,
    )
    print(f"Generated image saved to {output}")


if __name__ == "__main__":
    main()
