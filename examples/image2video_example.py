"""Image-to-video example using the Python API.

Generates a short video from an input image and text prompt using LTX 2.3 MLX.

Usage:
    uv run python examples/image2video_example.py

Requires an input image at input.jpg (or change INPUT_PATH below).
"""

from pathlib import Path

from image2video.client import generate

INPUT_PATH = Path("input.jpg")
OUTPUT_PATH = Path("output_image2video.mp4")


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"Create or place an image at {INPUT_PATH} to run this example.")
        return

    output = generate(
        image_path=INPUT_PATH,
        prompt="A cinematic drone shot slowly panning over the scene",
        output_path=OUTPUT_PATH,
        width=640,
        height=640,
        num_frames=81,
        num_inference_steps=4,
        fps=16,
        seed=42,
    )
    print(f"Generated video saved to {output}")


if __name__ == "__main__":
    main()
