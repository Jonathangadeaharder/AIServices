"""High-level Python API for LTX-2.3 video generation on MLX.

Provides simple functions for text-to-video, image-to-video, and audio-to-video
generation that wrap the underlying pipeline classes.

Usage:
    from ltx_pipelines_mlx.api import text_to_video, image_to_video, audio_to_video

    # Text-to-Video
    text_to_video(prompt="a cat walking", output_path="output.mp4")

    # Image-to-Video
    image_to_video(prompt="animate this", image="photo.jpg", output_path="output.mp4")

    # Audio-to-Video
    audio_to_video(prompt="music video", audio="music.wav", output_path="output.mp4")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "dgrauet/ltx-2.3-mlx-q8"
DEFAULT_GEMMA = "mlx-community/gemma-3-12b-it-4bit"


@dataclass
class VideoGenerationResult:
    """Result from a video generation operation."""

    output_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


def text_to_video(
    prompt: str,
    output_path: str,
    *,
    model: str = DEFAULT_MODEL,
    gemma: str = DEFAULT_GEMMA,
    height: int = 480,
    width: int = 704,
    num_frames: int = 97,
    seed: int = -1,
    num_steps: int | None = None,
    two_stage: bool = False,
    hq: bool = False,
    stage1_steps: int | None = None,
    stage2_steps: int | None = None,
    cfg_scale: float | None = None,
    stg_scale: float | None = None,
    enhance_prompt: bool = False,
    loras: list[tuple[str, float]] | None = None,
    quiet: bool = True,
) -> VideoGenerationResult:
    """Generate a video from a text prompt.

    Args:
        prompt: Text prompt describing the video to generate.
        output_path: Path to save the output video (.mp4).
        model: Model weights (HuggingFace repo ID or local path).
        gemma: Gemma model for text encoding.
        height: Video height in pixels.
        width: Video width in pixels.
        num_frames: Number of frames to generate.
        seed: Random seed (-1 for random).
        num_steps: Number of denoising steps (one-stage only).
        two_stage: Use two-stage pipeline for higher quality.
        hq: Use HQ two-stage pipeline (res_2s sampler).
        stage1_steps: Stage 1 denoising steps (two-stage only).
        stage2_steps: Stage 2 denoising steps (two-stage only).
        cfg_scale: CFG guidance scale.
        stg_scale: STG guidance scale.
        enhance_prompt: Enhance prompt using Gemma before generation.
        loras: List of (path, strength) tuples for LoRA weights.
        quiet: Suppress progress output.

    Returns:
        VideoGenerationResult with output path and metadata.
    """
    import random

    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    resolved_model = _resolve_model(model)

    if enhance_prompt:
        prompt = _enhance_prompt(prompt, gemma, seed=seed, is_image=False)

    lora_paths = loras or []

    if hq or two_stage:
        return _generate_two_stage(
            prompt=prompt,
            output_path=output_path,
            model=resolved_model,
            gemma=gemma,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            hq=hq,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            loras=lora_paths,
            quiet=quiet,
        )

    from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline

    if not quiet:
        print(f"Mode: Text-to-Video (one-stage, seed={seed})")

    pipe = TextToVideoPipeline(model_dir=resolved_model, gemma_model_id=gemma)
    if lora_paths:
        pipe._pending_loras = lora_paths

    pipe.generate_and_save(
        prompt=prompt,
        output_path=output_path,
        height=height,
        width=width,
        num_frames=num_frames,
        seed=seed,
        num_steps=num_steps,
    )

    return VideoGenerationResult(
        output_path=output_path,
        metadata={
            "provider": "ltx-pipelines-mlx",
            "mode": "text_to_video",
            "model": resolved_model,
            "seed": seed,
            "height": height,
            "width": width,
            "num_frames": num_frames,
        },
    )


def image_to_video(
    prompt: str,
    image: str,
    output_path: str,
    *,
    model: str = DEFAULT_MODEL,
    gemma: str = DEFAULT_GEMMA,
    height: int = 480,
    width: int = 704,
    num_frames: int = 97,
    seed: int = -1,
    num_steps: int | None = None,
    two_stage: bool = False,
    hq: bool = False,
    stage1_steps: int | None = None,
    stage2_steps: int | None = None,
    cfg_scale: float | None = None,
    stg_scale: float | None = None,
    enhance_prompt: bool = False,
    loras: list[tuple[str, float]] | None = None,
    quiet: bool = True,
) -> VideoGenerationResult:
    """Generate a video from a text prompt and reference image.

    Args:
        prompt: Text prompt describing the desired animation.
        image: Path to reference image file.
        output_path: Path to save the output video (.mp4).
        model: Model weights (HuggingFace repo ID or local path).
        gemma: Gemma model for text encoding.
        height: Video height in pixels.
        width: Video width in pixels.
        num_frames: Number of frames to generate.
        seed: Random seed (-1 for random).
        num_steps: Number of denoising steps (one-stage only).
        two_stage: Use two-stage pipeline for higher quality.
        hq: Use HQ two-stage pipeline (res_2s sampler).
        stage1_steps: Stage 1 denoising steps (two-stage only).
        stage2_steps: Stage 2 denoising steps (two-stage only).
        cfg_scale: CFG guidance scale.
        stg_scale: STG guidance scale.
        enhance_prompt: Enhance prompt using Gemma before generation.
        loras: List of (path, strength) tuples for LoRA weights.
        quiet: Suppress progress output.

    Returns:
        VideoGenerationResult with output path and metadata.
    """
    import random

    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    resolved_model = _resolve_model(model)

    if enhance_prompt:
        prompt = _enhance_prompt(prompt, gemma, seed=seed, is_image=True)

    lora_paths = loras or []

    if hq or two_stage:
        return _generate_two_stage(
            prompt=prompt,
            output_path=output_path,
            model=resolved_model,
            gemma=gemma,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            hq=hq,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            image=image,
            loras=lora_paths,
            quiet=quiet,
        )

    from ltx_pipelines_mlx.ti2vid_one_stage import ImageToVideoPipeline

    if not quiet:
        print(f"Mode: Image-to-Video (one-stage, seed={seed})")
        print(f"Image: {image}")

    pipe = ImageToVideoPipeline(model_dir=resolved_model, gemma_model_id=gemma)
    if lora_paths:
        pipe._pending_loras = lora_paths

    pipe.generate_and_save(
        prompt=prompt,
        output_path=output_path,
        image=image,
        height=height,
        width=width,
        num_frames=num_frames,
        seed=seed,
        num_steps=num_steps,
    )

    return VideoGenerationResult(
        output_path=output_path,
        metadata={
            "provider": "ltx-pipelines-mlx",
            "mode": "image_to_video",
            "model": resolved_model,
            "seed": seed,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "image": image,
        },
    )


def audio_to_video(
    prompt: str,
    audio: str,
    output_path: str,
    *,
    model: str = DEFAULT_MODEL,
    gemma: str = DEFAULT_GEMMA,
    height: int = 480,
    width: int = 704,
    num_frames: int = 97,
    fps: float = 24.0,
    seed: int = -1,
    hq: bool = False,
    stage1_steps: int | None = None,
    stage2_steps: int | None = None,
    cfg_scale: float | None = None,
    stg_scale: float | None = None,
    image: str | None = None,
    audio_start: float = 0.0,
    quiet: bool = True,
) -> VideoGenerationResult:
    """Generate a video from audio and text prompt.

    Args:
        prompt: Text prompt describing the video.
        audio: Path to input audio file (WAV/MP3/etc.).
        output_path: Path to save the output video (.mp4).
        model: Model weights (HuggingFace repo ID or local path).
        gemma: Gemma model for text encoding.
        height: Video height in pixels.
        width: Video width in pixels.
        num_frames: Number of frames to generate.
        fps: Frame rate.
        seed: Random seed (-1 for random).
        hq: Use HQ mode (res_2s sampler for stage 1).
        stage1_steps: Stage 1 denoising steps.
        stage2_steps: Stage 2 denoising steps.
        cfg_scale: CFG guidance scale.
        stg_scale: STG guidance scale.
        image: Optional reference image for I2V conditioning.
        audio_start: Audio start time in seconds.
        quiet: Suppress progress output.

    Returns:
        VideoGenerationResult with output path and metadata.
    """
    import random

    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    resolved_model = _resolve_model(model)

    if hq:
        from ltx_pipelines_mlx.a2vid_two_stage_hq import AudioToVideoHQPipeline as PipeClass

        mode_name = "Audio-to-Video HQ"
    else:
        from ltx_pipelines_mlx.a2vid_two_stage import AudioToVideoPipeline as PipeClass

        mode_name = "Audio-to-Video"

    if not quiet:
        print(f"Mode: {mode_name} (seed={seed})")
        print(f"Audio: {audio}")

    pipe = PipeClass(model_dir=resolved_model, gemma_model_id=gemma)

    kwargs: dict = {
        "prompt": prompt,
        "output_path": output_path,
        "audio_path": audio,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "fps": fps,
        "seed": seed,
        "image": image,
        "audio_start_time": audio_start,
    }
    if stage1_steps is not None:
        kwargs["stage1_steps"] = stage1_steps
    if stage2_steps is not None:
        kwargs["stage2_steps"] = stage2_steps
    if cfg_scale is not None:
        kwargs["cfg_scale"] = cfg_scale
    if stg_scale is not None:
        kwargs["stg_scale"] = stg_scale

    pipe.generate_and_save(**kwargs)

    return VideoGenerationResult(
        output_path=output_path,
        metadata={
            "provider": "ltx-pipelines-mlx",
            "mode": "audio_to_video",
            "model": resolved_model,
            "seed": seed,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "fps": fps,
            "audio": audio,
        },
    )


def retake(
    prompt: str,
    video: str,
    output_path: str,
    start_frame: int,
    end_frame: int,
    *,
    model: str = DEFAULT_MODEL,
    gemma: str = DEFAULT_GEMMA,
    seed: int = -1,
    num_steps: int | None = None,
    cfg_scale: float | None = None,
    stg_scale: float | None = None,
    regenerate_audio: bool = True,
    quiet: bool = True,
) -> VideoGenerationResult:
    """Regenerate a time segment of an existing video.

    Args:
        prompt: Text prompt for the regenerated segment.
        video: Path to source video file.
        output_path: Path to save the output video.
        start_frame: Start latent frame index (inclusive).
        end_frame: End latent frame index (exclusive).
        model: Model weights (HuggingFace repo ID or local path).
        gemma: Gemma model for text encoding.
        seed: Random seed (-1 for random).
        num_steps: Number of denoising steps.
        cfg_scale: CFG guidance scale.
        stg_scale: STG guidance scale.
        regenerate_audio: Whether to regenerate audio for the segment.
        quiet: Suppress progress output.

    Returns:
        VideoGenerationResult with output path and metadata.
    """
    import random

    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    resolved_model = _resolve_model(model)

    from ltx_pipelines_mlx.retake import RetakePipeline

    if not quiet:
        print(f"Mode: Retake (seed={seed})")
        print(f"Video: {video}, frames {start_frame}-{end_frame}")

    pipe = RetakePipeline(model_dir=resolved_model, gemma_model_id=gemma)

    kwargs: dict = {
        "prompt": prompt,
        "video_path": video,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "seed": seed,
        "regenerate_audio": regenerate_audio,
    }
    if num_steps is not None:
        kwargs["num_steps"] = num_steps
    if cfg_scale is not None:
        kwargs["cfg_scale"] = cfg_scale
    if stg_scale is not None:
        kwargs["stg_scale"] = stg_scale

    video_latent, audio_latent = pipe.retake_from_video(**kwargs)

    _decode_and_save(pipe, video_latent, audio_latent, output_path)

    return VideoGenerationResult(
        output_path=output_path,
        metadata={
            "provider": "ltx-pipelines-mlx",
            "mode": "retake",
            "model": resolved_model,
            "seed": seed,
            "start_frame": start_frame,
            "end_frame": end_frame,
        },
    )


def extend(
    prompt: str,
    video: str,
    output_path: str,
    extend_frames: int,
    *,
    direction: str = "after",
    model: str = DEFAULT_MODEL,
    gemma: str = DEFAULT_GEMMA,
    seed: int = -1,
    num_steps: int | None = None,
    cfg_scale: float | None = None,
    stg_scale: float | None = None,
    quiet: bool = True,
) -> VideoGenerationResult:
    """Add frames before or after an existing video.

    Args:
        prompt: Text prompt for the extended frames.
        video: Path to source video file.
        output_path: Path to save the output video.
        extend_frames: Number of latent frames to add.
        direction: Direction to extend ("before" or "after").
        model: Model weights (HuggingFace repo ID or local path).
        gemma: Gemma model for text encoding.
        seed: Random seed (-1 for random).
        num_steps: Number of denoising steps.
        cfg_scale: CFG guidance scale.
        stg_scale: STG guidance scale.
        quiet: Suppress progress output.

    Returns:
        VideoGenerationResult with output path and metadata.
    """
    import random

    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    resolved_model = _resolve_model(model)

    from ltx_pipelines_mlx.extend import ExtendPipeline

    if not quiet:
        print(f"Mode: Extend {direction} (seed={seed})")
        print(f"Video: {video}, +{extend_frames} latent frames")

    pipe = ExtendPipeline(model_dir=resolved_model, gemma_model_id=gemma)

    kwargs: dict = {
        "prompt": prompt,
        "video_path": video,
        "extend_frames": extend_frames,
        "direction": direction,
        "seed": seed,
    }
    if num_steps is not None:
        kwargs["num_steps"] = num_steps
    if cfg_scale is not None:
        kwargs["cfg_scale"] = cfg_scale
    if stg_scale is not None:
        kwargs["stg_scale"] = stg_scale

    video_latent, audio_latent = pipe.extend_from_video(**kwargs)

    _decode_and_save(pipe, video_latent, audio_latent, output_path)

    return VideoGenerationResult(
        output_path=output_path,
        metadata={
            "provider": "ltx-pipelines-mlx",
            "mode": "extend",
            "model": resolved_model,
            "seed": seed,
            "direction": direction,
            "extend_frames": extend_frames,
        },
    )


def keyframe_interpolation(
    prompt: str,
    start_image: str,
    end_image: str,
    output_path: str,
    *,
    model: str = DEFAULT_MODEL,
    gemma: str = DEFAULT_GEMMA,
    height: int = 480,
    width: int = 704,
    num_frames: int = 97,
    fps: float = 24.0,
    seed: int = -1,
    stage1_steps: int | None = None,
    stage2_steps: int | None = None,
    cfg_scale: float | None = None,
    stg_scale: float | None = None,
    quiet: bool = True,
) -> VideoGenerationResult:
    """Interpolate between two keyframe images to generate a video.

    Args:
        prompt: Text prompt describing the transition.
        start_image: Path to start keyframe image.
        end_image: Path to end keyframe image.
        output_path: Path to save the output video.
        model: Model weights (HuggingFace repo ID or local path).
        gemma: Gemma model for text encoding.
        height: Video height in pixels.
        width: Video width in pixels.
        num_frames: Number of frames to generate.
        fps: Frame rate.
        seed: Random seed (-1 for random).
        stage1_steps: Stage 1 denoising steps.
        stage2_steps: Stage 2 denoising steps.
        cfg_scale: CFG guidance scale.
        stg_scale: STG guidance scale.
        quiet: Suppress progress output.

    Returns:
        VideoGenerationResult with output path and metadata.
    """
    import random

    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    resolved_model = _resolve_model(model)

    from ltx_core_mlx.components.guiders import MultiModalGuiderParams

    from ltx_pipelines_mlx.keyframe_interpolation import KeyframeInterpolationPipeline

    if not quiet:
        print(f"Mode: Keyframe Interpolation (seed={seed})")
        print(f"Start: {start_image}, End: {end_image}")

    last_pixel_frame = num_frames - 1

    pipe = KeyframeInterpolationPipeline(model_dir=resolved_model, gemma_model_id=gemma)

    cfg = cfg_scale if cfg_scale is not None else 3.0
    stg = stg_scale if stg_scale is not None else 1.0
    video_gp = MultiModalGuiderParams(
        cfg_scale=cfg,
        stg_scale=stg,
        rescale_scale=0.7,
        modality_scale=3.0,
        stg_blocks=[28],
    )
    audio_gp = MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=stg,
        rescale_scale=0.7,
        modality_scale=3.0,
        stg_blocks=[28],
    )

    pipe.generate_and_save(
        prompt=prompt,
        output_path=output_path,
        keyframe_images=[start_image, end_image],
        keyframe_indices=[0, last_pixel_frame],
        height=height,
        width=width,
        num_frames=num_frames,
        fps=fps,
        seed=seed,
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        cfg_scale=cfg,
        video_guider_params=video_gp,
        audio_guider_params=audio_gp,
    )

    return VideoGenerationResult(
        output_path=output_path,
        metadata={
            "provider": "ltx-pipelines-mlx",
            "mode": "keyframe_interpolation",
            "model": resolved_model,
            "seed": seed,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "fps": fps,
        },
    )


# =============================================================================
# Internal helpers
# =============================================================================


def _resolve_model(model: str) -> str:
    """Resolve model path, downloading from HuggingFace if needed."""

    path = Path(model)
    if path.exists():
        return str(path)

    try:
        from huggingface_hub import snapshot_download

        return str(Path(snapshot_download(model)))
    except Exception:
        return model


def _enhance_prompt(prompt: str, gemma: str, seed: int, is_image: bool) -> str:
    """Enhance prompt using Gemma."""
    from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
    from ltx_core_mlx.utils.memory import aggressive_cleanup

    gemma_model = GemmaLanguageModel()
    gemma_model.load(gemma)
    if is_image:
        enhanced = gemma_model.enhance_i2v(prompt, seed=seed)
    else:
        enhanced = gemma_model.enhance_t2v(prompt, seed=seed)
    del gemma_model
    aggressive_cleanup()
    return enhanced


def _generate_two_stage(
    prompt: str,
    output_path: str,
    model: str,
    gemma: str,
    height: int,
    width: int,
    num_frames: int,
    seed: int,
    hq: bool,
    stage1_steps: int | None,
    stage2_steps: int | None,
    cfg_scale: float | None,
    stg_scale: float | None,
    image: str | None = None,
    loras: list[tuple[str, float]] | None = None,
    quiet: bool = True,
) -> VideoGenerationResult:
    """Generate using two-stage pipeline."""
    if hq:
        from ltx_pipelines_mlx.ti2vid_two_stages_hq import TwoStageHQPipeline as PipeClass

        mode_name = "HQ Two-Stage"
    else:
        from ltx_pipelines_mlx.ti2vid_two_stages import TwoStagePipeline as PipeClass

        mode_name = "Two-Stage"

    if not quiet:
        print(f"Mode: {mode_name} (seed={seed})")

    pipe = PipeClass(model_dir=model, gemma_model_id=gemma, low_memory=True)

    if loras:
        pipe._pending_loras = loras

    kwargs: dict = {
        "prompt": prompt,
        "output_path": output_path,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "seed": seed,
        "image": image,
    }
    if stage1_steps is not None:
        kwargs["stage1_steps"] = stage1_steps
    if stage2_steps is not None:
        kwargs["stage2_steps"] = stage2_steps
    if cfg_scale is not None:
        kwargs["cfg_scale"] = cfg_scale
    if stg_scale is not None:
        kwargs["stg_scale"] = stg_scale

    pipe.generate_and_save(**kwargs)

    return VideoGenerationResult(
        output_path=output_path,
        metadata={
            "provider": "ltx-pipelines-mlx",
            "mode": "two_stage" if not hq else "hq_two_stage",
            "model": model,
            "seed": seed,
            "height": height,
            "width": width,
            "num_frames": num_frames,
        },
    )


def _decode_and_save(
    pipe: object,
    video_latent: object,
    audio_latent: object,
    output_path: str,
) -> None:
    """Decode latents and save to file."""
    from ltx_core_mlx.utils.memory import aggressive_cleanup

    if hasattr(pipe, "low_memory") and pipe.low_memory:
        pipe.dit = None
        pipe.text_encoder = None
        pipe.feature_extractor = None
        pipe._loaded = False
        aggressive_cleanup()

    pipe._load_decoders()
    pipe._decode_and_save_video(video_latent, audio_latent, output_path)
