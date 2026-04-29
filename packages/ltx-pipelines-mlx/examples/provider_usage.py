"""Example usage of the LTX Video Provider with aiservices_core.

This script demonstrates how to use the provider pattern
for video generation.
"""

from aiservices_core.providers import registry
from ltx_pipelines_mlx.providers import LtxVideoProvider


def main():
    # Register the provider
    registry.register("video.ltx", LtxVideoProvider)

    # Get provider instance
    provider = registry.get("video.ltx")

    # Example 1: Text-to-Video
    result = provider.generate(
        prompt="A cat walking through a garden",
        output_path="outputs/provider_t2v.mp4",
        mode="t2v",
        height=480,
        width=704,
        num_frames=97,
        seed=42,
    )
    print(f"T2V Result: {result.output_path}")

    # Example 2: Image-to-Video
    result = provider.generate(
        prompt="The cat starts moving",
        output_path="outputs/provider_i2v.mp4",
        mode="i2v",
        image="inputs/cat.jpg",
        seed=42,
    )
    print(f"I2V Result: {result.output_path}")

    # Example 3: Audio-to-Video
    result = provider.generate(
        prompt="A dancer performing",
        output_path="outputs/provider_a2v.mp4",
        mode="a2v",
        audio="inputs/music.mp3",
        seed=42,
    )
    print(f"A2V Result: {result.output_path}")


if __name__ == "__main__":
    main()
