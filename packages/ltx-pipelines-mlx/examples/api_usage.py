"""Example usage of the ltx-pipelines-mlx Python API.

This script demonstrates how to use the high-level API functions
for video generation.
"""

from ltx_pipelines_mlx.api import (
    audio_to_video,
    image_to_video,
    text_to_video,
)


def example_text_to_video():
    """Generate a video from text only."""
    result = text_to_video(
        prompt="A cat walking gracefully through a sunny garden",
        output_path="outputs/cat_walking.mp4",
        height=480,
        width=704,
        num_frames=97,
        seed=42,
    )
    print(f"Generated: {result.output_path}")
    print(f"Metadata: {result.metadata}")


def example_image_to_video():
    """Generate a video from an image + text prompt."""
    result = image_to_video(
        prompt="The cat starts walking forward",
        image="inputs/cat_photo.jpg",
        output_path="outputs/cat_animated.mp4",
        height=480,
        width=704,
        num_frames=97,
        seed=42,
    )
    print(f"Generated: {result.output_path}")
    print(f"Metadata: {result.metadata}")


def example_audio_to_video():
    """Generate a video synchronized to audio."""
    result = audio_to_video(
        prompt="A dancer performing to the music",
        audio="inputs/music.mp3",
        output_path="outputs/dancer.mp4",
        height=480,
        width=704,
        num_frames=97,
        fps=24.0,
        seed=42,
    )
    print(f"Generated: {result.output_path}")
    print(f"Metadata: {result.metadata}")


def example_two_stage():
    """Generate a higher quality video using two-stage pipeline."""
    result = text_to_video(
        prompt="A beautiful sunset over the ocean",
        output_path="outputs/sunset_hq.mp4",
        height=480,
        width=704,
        num_frames=97,
        seed=42,
        two_stage=True,  # Use two-stage for better quality
        stage1_steps=30,
        stage2_steps=3,
    )
    print(f"Generated: {result.output_path}")


def example_with_loras():
    """Generate a video with custom LoRA weights."""
    result = text_to_video(
        prompt="A cinematic scene in a forest",
        output_path="outputs/forest_styled.mp4",
        height=480,
        width=704,
        num_frames=97,
        seed=42,
        loras=[
            ("path/to/cinematic_lora.safetensors", 0.8),
        ],
    )
    print(f"Generated: {result.output_path}")


def example_enhanced_prompt():
    """Generate a video with automatic prompt enhancement."""
    result = text_to_video(
        prompt="cat walking",
        output_path="outputs/cat_enhanced.mp4",
        enhance_prompt=True,  # Uses Gemma to enhance the prompt
        seed=42,
    )
    print(f"Generated: {result.output_path}")


if __name__ == "__main__":
    # Run the example you want
    import sys

    examples = {
        "t2v": example_text_to_video,
        "i2v": example_image_to_video,
        "a2v": example_audio_to_video,
        "hq": example_two_stage,
        "lora": example_with_loras,
        "enhance": example_enhanced_prompt,
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("Usage: python examples/api_usage.py <example>")
        print(f"Available examples: {', '.join(examples.keys())}")
