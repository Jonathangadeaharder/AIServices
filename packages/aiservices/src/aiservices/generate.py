"""Image, audio, and video generation abstraction layer.

Provides unified API for:
- text2image: Generate from prompts using FLUX.2 Klein (mflux)
- image2image: Edit existing images using FLUX.2 Klein (mflux)
- train_lora: LoRA fine-tuning for Flux models (mflux-train)
- text2audio: Generate speech from text via Fish Speech (mlx_audio)
- video generation: multiple LTX-2.3 pipelines including keyframe interpolation

All image/LoRA operations use mflux CLI tools via subprocess.
Video operations use ltx-pipelines-mlx Python API.
Audio operations use mlx_audio Python API.
"""

from __future__ import annotations

import subprocess
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

try:
    from structlog import get_logger
except ImportError:

    class _Log:
        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

    get_logger = lambda n: _Log()

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration & data classes
# ---------------------------------------------------------------------------


@dataclass
class GenConfig:
    """Configuration for image generation."""

    seed: int = 42
    steps: int = 4  # FLUX.2 Klein distilled: 4 steps optimal
    quality: Literal["draft", "standard", "high"] = "standard"


@dataclass
class LoRAAdapter:
    """A LoRA adapter with path and inference strength.

    Encapsulates a trained LoRA weight file and the strength at which
    to apply it during inference. Usable with both ImageGenerator (Flux)
    and VideoGenerator (LTX 2.3 IC-LoRA).

    Usage::

        adapter = LoRAAdapter(path="my_style.safetensors", strength=0.8)
        gen = ImageGenerator().with_lora(adapter)
        img = gen.text2image("a portrait in my style")

    Args:
        path: Path to the LoRA .safetensors file (local or HuggingFace repo ID).
        strength: LoRA inference strength in [0, 1]. 1.0 = full effect.
    """

    path: str
    strength: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"LoRA strength must be in [0, 1], got {self.strength}")

    @staticmethod
    def from_pairs(pairs: list[tuple[str, float]]) -> list[LoRAAdapter]:
        """Create a list of LoRAAdapter from (path, strength) tuples."""
        return [LoRAAdapter(path=p, strength=s) for p, s in pairs]

    def to_pair(self) -> tuple[str, float]:
        """Convert back to (path, strength) tuple for pipeline APIs."""
        return (self.path, self.strength)


class ImageFrame:
    """Represents a generated or edited image with metadata."""

    def __init__(
        self,
        path: Path,
        prompt: str,
        seed: int,
        width: int | None = None,
        height: int | None = None,
    ):
        self.path = path
        self.prompt = prompt
        self.seed = seed
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return f"ImageFrame(path={self.path}, seed={self.seed})"

    def __str__(self) -> str:
        return f"ImageFrame({self.path})"


# ---------------------------------------------------------------------------
# Video mode enum
# ---------------------------------------------------------------------------


class VideoMode(str, Enum):
    """Supported video generation modes.

    Each mode maps to a different ltx-pipelines-mlx pipeline class.
    """

    KEYFRAME = "keyframe"
    TWO_STAGE = "two-stage"
    HQ = "hq"
    DISTILLED = "distilled"
    ONE_STAGE = "one-stage"
    IC_LORA = "ic-lora"
    HDR_IC_LORA = "hdr-ic-lora"


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


class ImageGenerator:
    """Unified image generation interface.

    Wraps mflux CLI tools with a consistent API.
    Handles text-to-image, image-to-image, and LoRA training.

    Usage:
        gen = ImageGenerator()
        img = gen.text2image("a woodcut portrait of Hansel")
        edited = gen.image2image("add a hat", img.path)
    """

    MODEL = "mlx-community/FLUX.2-klein-9B"
    BASE_MODEL = "flux2-klein-9b"

    def __init__(self, config: GenConfig | None = None):
        self.config = config or GenConfig()
        self._lora_adapters: list[LoRAAdapter] = []

    def _resolve_lora(
        self,
        lora_paths: list[Path] | None = None,
        lora_scales: list[float] | None = None,
    ) -> tuple[list[Path] | None, list[float] | None]:
        """Merge explicit lora args with any adapters from with_lora()."""
        if not self._lora_adapters:
            return lora_paths, lora_scales
        adapter_paths = [Path(a.path) for a in self._lora_adapters]
        adapter_scales = [a.strength for a in self._lora_adapters]
        if lora_paths:
            adapter_paths.extend(lora_paths)
            adapter_scales.extend(lora_scales or [1.0] * len(lora_paths))
        return adapter_paths, adapter_scales

    def text2image(
        self,
        prompt: str,
        out_path: Path | None = None,
        lora_paths: list[Path] | None = None,
        lora_scales: list[float] | None = None,
    ) -> ImageFrame:
        """Generate image from text prompt using FLUX.2 Klein."""
        out_path = out_path or Path("output") / "text2image.png"
        paths, scales = self._resolve_lora(lora_paths, lora_scales)
        return _text2image(
            prompt=prompt,
            out_path=out_path,
            seed=self.config.seed,
            steps=self.config.steps,
            lora_paths=paths,
            lora_scales=scales,
        )

    def image2image(
        self,
        prompt: str,
        base_image: Path,
        out_path: Path | None = None,
        lora_paths: list[Path] | None = None,
        lora_scales: list[float] | None = None,
    ) -> ImageFrame:
        """Edit image using FLUX.2 Klein with prompt conditioning."""
        out_path = out_path or Path("output") / "image2image.png"
        paths, scales = self._resolve_lora(lora_paths, lora_scales)
        return _image2image(
            prompt=prompt,
            image_paths=[base_image],
            out_path=out_path,
            seed=self.config.seed,
            steps=self.config.steps,
            lora_paths=paths,
            lora_scales=scales,
        )

    def generate(
        self,
        prompt: str | None = None,
        base_image: Path | None = None,
        out_path: Path | None = None,
        lora_paths: list[Path] | None = None,
        lora_scales: list[float] | None = None,
    ) -> ImageFrame:
        """Auto-detect mode: text2image or image2image."""
        if base_image is not None:
            return self.image2image(
                prompt or "",
                base_image,
                out_path,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )
        if prompt is not None:
            return self.text2image(
                prompt,
                out_path,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )
        raise ValueError("Must provide either prompt or base_image")

    def train_lora(
        self,
        training_dir: Path | str,
        output_dir: Path | str,
        *,
        config_path: Path | str | None = None,
        resume_path: Path | str | None = None,
        dry_run: bool = False,
    ) -> Path:
        """Train a LoRA adapter for Flux using mflux-train.

        Wraps the ``mflux-train`` CLI with a simple interface.

        Args:
            training_dir: Directory containing training images + captions.
                Expected layout::

                    training_dir/
                      image_001.png
                      image_001.txt   # caption for image_001.png
                      image_002.png
                      image_002.txt

            output_dir: Directory where the trained adapter will be saved.
            config_path: Optional TOML config file. When provided, mflux-train
                uses it directly. When omitted, a default config is generated
                with sensible parameters for FLUX.2 Klein LoRA training.
            resume_path: Optional path to a checkpoint to resume training from.
            dry_run: If True, validate the config without training.

        Returns:
            Path to the output directory containing the trained adapter.
        """
        training_dir = Path(training_dir)
        output_dir = Path(output_dir)

        if not training_dir.is_dir():
            raise FileNotFoundError(f"Training directory not found: {training_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        if config_path is not None:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            config_arg = str(config_path)
        else:
            # Generate a minimal config with sensible defaults
            config_arg = _generate_lora_config(training_dir, output_dir)

        cmd = ["uv", "run", "mflux-train", "--config", config_arg]

        if resume_path is not None:
            cmd.extend(["--resume", str(resume_path)])

        if dry_run:
            cmd.append("--dry-run")

        logger.info(
            "lora_train_start",
            training_dir=str(training_dir),
            output_dir=str(output_dir),
            dry_run=dry_run,
        )

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"mflux-train failed:\n{result.stderr[-800:]}"
            )

        logger.info("lora_train_complete", output=str(output_dir))
        return output_dir

    def with_lora(
        self,
        *adapters: LoRAAdapter | tuple[str, float],
    ) -> ImageGenerator:
        """Return a new ImageGenerator with LoRA adapters pre-configured.

        Creates a copy of this generator with LoRA paths and scales set
        so every subsequent text2image/image2image call applies them
        automatically.

        Args:
            adapters: LoRAAdapter instances or (path, strength) tuples.

        Returns:
            New ImageGenerator with LoRA adapters applied.

        Usage::

            style = LoRAAdapter("my_style.safetensors", 0.8)
            gen = ImageGenerator().with_lora(style)
            img = gen.text2image("a portrait in my style")
        """
        resolved: list[LoRAAdapter] = []
        for a in adapters:
            if isinstance(a, LoRAAdapter):
                resolved.append(a)
            elif isinstance(a, tuple):
                resolved.append(LoRAAdapter(path=a[0], strength=a[1]))
            else:
                raise TypeError(f"Expected LoRAAdapter or tuple, got {type(a)}")

        new_gen = ImageGenerator(GenConfig(seed=self.config.seed, steps=self.config.steps))
        new_gen._lora_adapters = resolved
        return new_gen


def _generate_lora_config(training_dir: Path, output_dir: Path) -> str:
    """Generate a minimal mflux LoRA training config file.

    Returns path to the generated TOML config.
    """
    config_content = f"""\
[model]
name = "mlx-community/FLUX.2-klein-9B"
base_model = "flux2-klein-9b"

[training]
images_path = "{training_dir}"
output_path = "{output_dir}"
steps = 1000
rank = 16
learning_rate = 1e-4
"""
    config_file = Path(tempfile.mktemp(suffix=".toml", prefix="mflux_lora_"))
    config_file.write_text(config_content, encoding="utf-8")
    return str(config_file)


def _text2image(
    prompt: str,
    out_path: Path,
    seed: int = 42,
    steps: int = 4,
    lora_paths: list[Path] | None = None,
    lora_scales: list[float] | None = None,
) -> ImageFrame:
    """Generate image from text using mflux-generate-flux2 CLI."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s = steps or 4

    cmd = [
        "uv",
        "run",
        "mflux-generate-flux2",
        "--model",
        "mlx-community/FLUX.2-klein-9B",
        "--base-model",
        "flux2-klein-9b",
        "--prompt",
        prompt,
        "--output",
        str(out_path),
        "--seed",
        str(seed),
        "--steps",
        str(s),
    ]

    if lora_paths:
        cmd.extend(["--lora-paths", *(str(p) for p in lora_paths)])
    if lora_scales:
        cmd.extend(["--lora-scales", *(str(s) for s in lora_scales)])

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"mflux-generate-flux2 failed (seed={seed}):\n{result.stderr[-800:]}"
        )
    if not out_path.exists():
        raise RuntimeError(f"mflux-generate-flux2 produced no output at {out_path}")

    logger.info("text2image_complete", path=str(out_path), seed=seed, steps=s)
    return ImageFrame(path=out_path, prompt=prompt, seed=seed, width=1024, height=1024)


def _image2image(
    prompt: str,
    image_paths: list[Path],
    out_path: Path,
    seed: int = 42,
    steps: int = 4,
    lora_paths: list[Path] | None = None,
    lora_scales: list[float] | None = None,
) -> ImageFrame:
    """Edit image using mflux-generate-flux2-edit CLI."""
    if not image_paths:
        raise ValueError("image2image requires at least one image path")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "mflux-generate-flux2-edit",
        "--model",
        "mlx-community/FLUX.2-klein-9B",
        "--base-model",
        "flux2-klein-9b",
        "--prompt",
        prompt,
        "--output",
        str(out_path),
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--image-paths",
        *(str(p) for p in image_paths),
    ]

    if lora_paths:
        cmd.extend(["--lora-paths", *(str(p) for p in lora_paths)])
    if lora_scales:
        cmd.extend(["--lora-scales", *(str(s) for s in lora_scales)])

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"mflux-generate-flux2-edit failed (seed={seed}):\n{result.stderr[-800:]}"
        )
    if not out_path.exists():
        raise RuntimeError(
            f"mflux-generate-flux2-edit produced no output at {out_path}"
        )

    logger.info("image2image_complete", path=str(out_path), seed=seed, steps=steps)
    return ImageFrame(
        path=out_path, prompt=prompt, seed=seed, width=1024, height=1024
    )


# ---------------------------------------------------------------------------
# Convenience functions — image
# ---------------------------------------------------------------------------


def generate_image(
    prompt: str | None = None,
    base_image: Path | None = None,
    out_path: Path | None = None,
    seed: int = 42,
    steps: int = 4,
    lora_paths: list[Path] | None = None,
    lora_scales: list[float] | None = None,
) -> ImageFrame:
    """Convenience function for image generation."""
    gen = ImageGenerator(GenConfig(seed=seed, steps=steps))
    return gen.generate(
        prompt=prompt,
        base_image=base_image,
        out_path=out_path,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
    )


def generate_text2image(
    prompt: str,
    out_path: Path | None = None,
    seed: int = 42,
    steps: int = 4,
    lora_paths: list[Path] | None = None,
    lora_scales: list[float] | None = None,
) -> ImageFrame:
    """Generate image from text prompt using FLUX.2 Klein."""
    return _text2image(
        prompt=prompt,
        out_path=out_path or Path("output"),
        seed=seed,
        steps=steps,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
    )


def generate_image2image(
    prompt: str,
    base_image: Path,
    out_path: Path | None = None,
    seed: int = 42,
    steps: int = 4,
    lora_paths: list[Path] | None = None,
    lora_scales: list[float] | None = None,
) -> ImageFrame:
    """Edit image with prompt guidance using FLUX.2 Klein."""
    return _image2image(
        prompt=prompt,
        image_paths=[base_image],
        out_path=out_path or Path("output"),
        seed=seed,
        steps=steps,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
    )


# ---------------------------------------------------------------------------
# Audio generation — Fish Speech via mlx_audio
# ---------------------------------------------------------------------------


class AudioClip:
    """Represents a generated audio clip."""

    def __init__(self, path: Path, duration_s: float, sample_rate: int = 24000):
        self.path = path
        self.duration_s = duration_s
        self.sample_rate = sample_rate

    def __repr__(self) -> str:
        return f"AudioClip(path={self.path}, duration={self.duration_s:.1f}s)"


class AudioGenerator:
    """Text-to-speech via Fish Speech (mlx_audio).

    Uses the modern mlx_audio load_model API for speech generation.
    Supports reference audio for voice cloning.

    Usage:
        gen = AudioGenerator()
        path = gen.generate("Hello world", output="/tmp/out.wav")
    """

    DEFAULT_MODEL = "fishaudio/fish-speech-1.5"

    def __init__(
        self,
        seed: int = 42,
        temperature: float = 0.8,
        model: str | None = None,
    ):
        self.seed = seed
        self.temperature = temperature
        self.model = model or self.DEFAULT_MODEL

    def generate(
        self,
        text: str,
        output: str | Path,
        *,
        file_prefix: str = "audio",
        audio_format: str = "wav",
        ref_audio: str | Path | None = None,
        ref_text: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> Path:
        """Generate speech from text using Fish Speech.

        Args:
            text: Text to synthesize
            output: Output directory (when file_prefix given) or full file path
            file_prefix: Prefix for output files (enables numbered output mode)
            audio_format: Output format (wav, etc.)
            ref_audio: Reference audio path for voice cloning
            ref_text: Reference text matching ref_audio
            temperature: Generation temperature (overrides instance default)
            seed: Random seed (overrides instance default)

        Returns:
            Path to generated audio file
        """
        out = Path(output)

        actual_seed = seed if seed is not None else self.seed
        actual_temp = temperature if temperature is not None else self.temperature

        try:
            from mlx_audio.tts.generate import generate_audio as _gen_audio  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "mlx-audio not installed (pip install mlx-audio)"
            ) from e

        # Directory output with prefix: mlx_audio writes {prefix}_000.wav etc.
        if file_prefix:
            out.mkdir(parents=True, exist_ok=True)
            _gen_audio(
                text=text,
                model=self.model,
                output_path=str(out),
                file_prefix=file_prefix,
                audio_format=audio_format,
                reference_audio=str(ref_audio) if ref_audio else None,
                ref_text=ref_text,
                seed=actual_seed,
                temperature=actual_temp,
            )
            expected = out / f"{file_prefix}_000.{audio_format}"
            if not expected.exists():
                raise RuntimeError(
                    f"Audio generation produced no output at {expected}"
                )
            return expected

        # Single file mode
        out.parent.mkdir(parents=True, exist_ok=True)
        _gen_audio(
            text=text,
            model=self.model,
            output_path=str(out),
            reference_audio=str(ref_audio) if ref_audio else None,
            ref_text=ref_text,
            seed=actual_seed,
            temperature=actual_temp,
        )
        if not out.exists():
            raise RuntimeError(f"Audio generation produced no output at {out}")
        return out


def generate_audio(
    text: str,
    output: str | Path,
    *,
    ref_audio: str | Path | None = None,
    ref_text: str | None = None,
    seed: int = 42,
    temperature: float = 0.8,
) -> Path:
    """Convenience function for text-to-speech."""
    gen = AudioGenerator(seed=seed, temperature=temperature)
    return gen.generate(text, output, file_prefix="", ref_audio=ref_audio, ref_text=ref_text)


# ---------------------------------------------------------------------------
# Video generation — LTX-2.3 via ltx-pipelines-mlx
# ---------------------------------------------------------------------------


class VideoGenerator:
    """Video generation via LTX-2.3 MLX pipelines.

    Supports seven pipeline modes:
    - ``keyframe``: Interpolation between start+end frame images
    - ``two-stage``: Recommended general-purpose I2V/T2V (dev model + CFG + upscale)
    - ``hq``: Highest quality (res_2s sampler + CFG + upscale)
    - ``distilled``: Fastest (distilled half-res + upscale)
    - ``one-stage``: Full-resolution CFG, no upscaler dependency
    - ``ic-lora``: IC-LoRA conditioned generation with reference video
    - ``hdr-ic-lora``: HDR IC-LoRA generation (SDR mp4 + HDR .npz)

    LoRA support:
    - ``with_lora()`` returns a new VideoGenerator with IC-LoRA adapters
    - ``train_lora()`` trains a LoRA adapter using ltx-2-mlx train CLI
    - ``preprocess_training_data()`` preprocesses videos for training

    Usage:
        gen = VideoGenerator()

        # Keyframe interpolation (start + end frame)
        path = gen.generate(
            prompt="smooth transition between scenes",
            output="transition.mp4",
            mode="keyframe",
            start_image="frame1.png",
            end_image="frame2.png",
        )

        # IC-LoRA with reference video conditioning
        adapter = LoRAAdapter("Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control", 1.0)
        gen_lora = gen.with_lora(adapter)
        path = gen_lora.generate(
            prompt="person dancing",
            output="dance.mp4",
            mode="ic-lora",
            video_conditioning=[("depth_map.mp4", 1.0)],
        )
    """

    DEFAULT_MODEL = "dgrauet/ltx-2.3-mlx-q8"

    def __init__(
        self,
        model_dir: str | None = None,
        seed: int = 42,
        fps: int = 24,
        low_memory: bool = True,
    ):
        self.model_dir = model_dir or self.DEFAULT_MODEL
        self.seed = seed
        self.fps = fps
        self.low_memory = low_memory
        self._lora_adapters: list[LoRAAdapter] = []

    def with_lora(
        self,
        *adapters: LoRAAdapter | tuple[str, float],
    ) -> VideoGenerator:
        """Return a new VideoGenerator with IC-LoRA adapters pre-configured.

        The returned generator auto-applies LoRA to ic-lora and hdr-ic-lora modes.

        Args:
            adapters: LoRAAdapter instances or (path, strength) tuples.

        Returns:
            New VideoGenerator with LoRA adapters applied.
        """
        resolved: list[LoRAAdapter] = []
        for a in adapters:
            if isinstance(a, LoRAAdapter):
                resolved.append(a)
            elif isinstance(a, tuple):
                resolved.append(LoRAAdapter(path=a[0], strength=a[1]))
            else:
                raise TypeError(f"Expected LoRAAdapter or tuple, got {type(a)}")

        new_gen = VideoGenerator(
            model_dir=self.model_dir,
            seed=self.seed,
            fps=self.fps,
            low_memory=self.low_memory,
        )
        new_gen._lora_adapters = list(self._lora_adapters) + resolved
        return new_gen

    def generate(
        self,
        prompt: str,
        output: str | Path,
        *,
        mode: VideoMode | str = "two-stage",
        image: str | Path | None = None,
        start_image: str | Path | None = None,
        end_image: str | Path | None = None,
        duration: float | None = None,
        fps: int | None = None,
        width: int = 704,
        height: int = 480,
        num_frames: int | None = None,
        seed: int | None = None,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        cfg_scale: float = 3.0,
        dev_transformer: str | None = None,
        distilled_lora: str | None = None,
        lora_paths: list[LoRAAdapter | tuple[str, float]] | None = None,
        video_conditioning: list[tuple[str, float]] | None = None,
        conditioning_strength: float = 1.0,
        skip_stage_2: bool = False,
    ) -> Path:
        """Generate video from text or images using the specified LTX-2.3 pipeline.

        Args:
            prompt: Text prompt describing motion/content
            output: Output video file path
            mode: Pipeline mode. One of VideoMode values.
            image: Source image for I2V conditioning
            start_image: Start keyframe image (keyframe mode)
            end_image: End keyframe image (keyframe mode)
            duration: Desired duration in seconds (overrides num_frames)
            fps: Frames per second (overrides instance default)
            width: Video width
            height: Video height
            num_frames: Total pixel frames (must be 8k+1). Common: 97=4s, 121=5s
            seed: Random seed (overrides instance default)
            stage1_steps: Stage 1 denoising steps
            stage2_steps: Stage 2 denoising steps
            cfg_scale: CFG guidance scale (1.0 = no guidance, 3.0 = standard)
            dev_transformer: Dev transformer filename for keyframe mode
            distilled_lora: Distilled LoRA filename for keyframe mode
            lora_paths: IC-LoRA adapters (overrides with_lora() adapters).
                Accepts LoRAAdapter instances or (path, strength) tuples.
            video_conditioning: Reference control videos for IC-LoRA as
                (video_path, strength) tuples (e.g., depth maps, poses).
            conditioning_strength: IC-LoRA attention strength in [0, 1].
            skip_stage_2: Skip upscale + refine (half-res output).

        Returns:
            Path to generated video file
        """
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)

        actual_seed = seed if seed is not None else self.seed
        actual_fps = fps if fps is not None else self.fps

        if num_frames is None and duration is not None:
            num_frames = int(duration * actual_fps)
        if num_frames is None:
            num_frames = 97  # ~4s at 24fps

        # Normalize mode
        if isinstance(mode, str):
            mode = VideoMode(mode)

        # Auto-detect keyframe mode
        if start_image is not None and end_image is not None and mode != VideoMode.KEYFRAME:
            mode = VideoMode.KEYFRAME

        if mode == VideoMode.KEYFRAME and (start_image is None or end_image is None):
            raise ValueError("Keyframe mode requires both start_image and end_image")

        if mode == VideoMode.KEYFRAME:
            if dev_transformer is None:
                dev_transformer = "transformer-dev.safetensors"
            if distilled_lora is None:
                distilled_lora = "ltx-2.3-22b-distilled-lora-384.safetensors"

        # Resolve LoRA adapters: explicit arg > with_lora() > empty
        resolved_lora = self._resolve_video_lora(lora_paths)

        # Auto-detect ic-lora mode when lora_paths + video_conditioning given
        if resolved_lora and video_conditioning and mode not in (VideoMode.IC_LORA, VideoMode.HDR_IC_LORA):
            mode = VideoMode.IC_LORA

        if mode in (VideoMode.IC_LORA, VideoMode.HDR_IC_LORA) and not resolved_lora:
            raise ValueError(f"{mode.value} mode requires lora_paths (IC-LoRA adapters)")

        if mode == VideoMode.IC_LORA and not video_conditioning:
            raise ValueError("ic-lora mode requires video_conditioning (reference control videos)")

        return _run_video_pipeline(
            mode=mode,
            prompt=prompt,
            output_path=str(out),
            model_dir=self.model_dir,
            image=str(image) if image else None,
            start_image=str(start_image) if start_image else None,
            end_image=str(end_image) if end_image else None,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=float(actual_fps),
            seed=actual_seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            cfg_scale=cfg_scale,
            low_memory=self.low_memory,
            dev_transformer=dev_transformer,
            distilled_lora=distilled_lora,
            lora_paths=resolved_lora,
            video_conditioning=video_conditioning or [],
            conditioning_strength=conditioning_strength,
            skip_stage_2=skip_stage_2,
        )

    def _resolve_video_lora(
        self,
        lora_paths: list[LoRAAdapter | tuple[str, float]] | None = None,
    ) -> list[tuple[str, float]]:
        """Merge explicit lora_paths with any adapters from with_lora()."""
        result: list[tuple[str, float]] = []
        if lora_paths:
            for item in lora_paths:
                if isinstance(item, LoRAAdapter):
                    result.append(item.to_pair())
                elif isinstance(item, tuple):
                    result.append(item)
        elif self._lora_adapters:
            result = [a.to_pair() for a in self._lora_adapters]
        return result

    def train_lora(
        self,
        config_path: Path | str,
        *,
        videos_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
    ) -> Path:
        """Train a LoRA adapter for LTX-2.3 using ltx-2-mlx train.

        Wraps the ``ltx-2-mlx train`` CLI. Requires a YAML training config
        and optionally a preprocessed dataset.

        Args:
            config_path: Path to YAML training config file.
                See ltx-2-mlx docs for config format.
            videos_dir: Optional directory of training videos.
            output_dir: Optional output directory for trained weights.

        Returns:
            Path to the output directory containing the trained adapter.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Preprocess if videos_dir provided
        if videos_dir is not None:
            self.preprocess_training_data(
                videos_dir=videos_dir,
                output_dir=output_dir or Path("preprocessed"),
            )

        cmd = [
            "uv", "run", "ltx-2-mlx", "train",
            "--config", str(config_path),
        ]

        logger.info("video_lora_train_start", config=str(config_path))

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"ltx-2-mlx train failed:\n{result.stderr[-800:]}"
            )

        output = Path(output_dir) if output_dir else Path("output")
        logger.info("video_lora_train_complete", output=str(output))
        return output

    def preprocess_training_data(
        self,
        videos_dir: Path | str,
        output_dir: Path | str,
        *,
        captions_dir: Path | str | None = None,
        max_frames: int = 97,
        target_height: int | None = None,
        target_width: int | None = None,
    ) -> Path:
        """Preprocess videos into latents for LTX-2.3 LoRA training.

        Wraps ``ltx-2-mlx preprocess`` CLI.

        Args:
            videos_dir: Directory containing training video files.
            output_dir: Directory for preprocessed latents.
            captions_dir: Optional directory with .txt caption files.
            max_frames: Max frames per video (must satisfy 8k+1).
            target_height: Resize height (default: keep original).
            target_width: Resize width (default: keep original).

        Returns:
            Path to output directory with preprocessed data.
        """
        videos_dir = Path(videos_dir)
        output_dir = Path(output_dir)

        if not videos_dir.is_dir():
            raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "uv", "run", "ltx-2-mlx", "preprocess",
            "--videos", str(videos_dir),
            "--output", str(output_dir),
            "--model", self.model_dir,
            "--max-frames", str(max_frames),
        ]

        if captions_dir:
            cmd.extend(["--captions", str(captions_dir)])
        if target_height:
            cmd.extend(["--height", str(target_height)])
        if target_width:
            cmd.extend(["--width", str(target_width)])

        logger.info(
            "video_preprocess_start",
            videos=str(videos_dir),
            output=str(output_dir),
        )

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"ltx-2-mlx preprocess failed:\n{result.stderr[-800:]}"
            )

        logger.info("video_preprocess_complete", output=str(output_dir))
        return output_dir


def _run_video_pipeline(
    *,
    mode: VideoMode,
    prompt: str,
    output_path: str,
    model_dir: str,
    image: str | None,
    start_image: str | None,
    end_image: str | None,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    seed: int,
    stage1_steps: int | None,
    stage2_steps: int | None,
    cfg_scale: float,
    low_memory: bool,
    dev_transformer: str | None,
    distilled_lora: str | None,
    lora_paths: list[tuple[str, float]] | None = None,
    video_conditioning: list[tuple[str, float]] | None = None,
    conditioning_strength: float = 1.0,
    skip_stage_2: bool = False,
) -> Path:
    """Dispatch to the correct ltx-pipelines-mlx pipeline class."""
    try:
        import ltx_pipelines_mlx  # type: ignore[import]  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "ltx-pipelines-mlx not installed. "
            "Install from: https://github.com/dgrauet/ltx-2-mlx"
        ) from e

    output = Path(output_path)

    if mode == VideoMode.KEYFRAME:
        return _run_keyframe_pipeline(
            prompt=prompt,
            output_path=output,
            model_dir=model_dir,
            start_image=start_image,
            end_image=end_image,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            cfg_scale=cfg_scale,
            low_memory=low_memory,
            dev_transformer=dev_transformer,
            distilled_lora=distilled_lora,
        )
    elif mode == VideoMode.TWO_STAGE:
        return _run_two_stage_pipeline(
            prompt=prompt,
            output_path=output,
            model_dir=model_dir,
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            cfg_scale=cfg_scale,
            low_memory=low_memory,
        )
    elif mode == VideoMode.HQ:
        return _run_hq_pipeline(
            prompt=prompt,
            output_path=output,
            model_dir=model_dir,
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            cfg_scale=cfg_scale,
            low_memory=low_memory,
        )
    elif mode == VideoMode.DISTILLED:
        return _run_distilled_pipeline(
            prompt=prompt,
            output_path=output,
            model_dir=model_dir,
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            low_memory=low_memory,
        )
    elif mode == VideoMode.ONE_STAGE:
        return _run_one_stage_pipeline(
            prompt=prompt,
            output_path=output,
            model_dir=model_dir,
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            cfg_scale=cfg_scale,
            low_memory=low_memory,
        )
    elif mode == VideoMode.IC_LORA:
        return _run_ic_lora_pipeline(
            prompt=prompt,
            output_path=output,
            model_dir=model_dir,
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            low_memory=low_memory,
            lora_paths=lora_paths or [],
            video_conditioning=video_conditioning or [],
            conditioning_strength=conditioning_strength,
            skip_stage_2=skip_stage_2,
        )
    elif mode == VideoMode.HDR_IC_LORA:
        return _run_hdr_ic_lora_pipeline(
            prompt=prompt,
            output_path=output,
            model_dir=model_dir,
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            low_memory=low_memory,
            lora_paths=lora_paths or [],
            video_conditioning=video_conditioning or [],
            conditioning_strength=conditioning_strength,
            skip_stage_2=skip_stage_2,
        )
    else:
        raise ValueError(f"Unknown video mode: {mode!r}")


def _run_keyframe_pipeline(
    *,
    prompt: str,
    output_path: Path,
    model_dir: str,
    start_image: str | None,
    end_image: str | None,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    seed: int,
    stage1_steps: int | None,
    stage2_steps: int | None,
    cfg_scale: float,
    low_memory: bool,
    dev_transformer: str | None,
    distilled_lora: str | None,
) -> Path:
    """Keyframe interpolation between start and end images."""
    from ltx_pipelines_mlx import KeyframeInterpolationPipeline

    logger.info(
        "video_keyframe_start",
        model=model_dir,
        start=start_image,
        end=end_image,
        frames=num_frames,
    )

    pipe = KeyframeInterpolationPipeline(
        model_dir=model_dir,
        low_memory=low_memory,
        dev_transformer=dev_transformer,
        distilled_lora=distilled_lora,
    )

    # Keyframe indices: start at frame 0, end at last frame
    pipe.generate_and_save(
        prompt=prompt,
        output_path=str(output_path),
        keyframe_images=[start_image, end_image],
        keyframe_indices=[0, num_frames - 1],
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        seed=seed,
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        cfg_scale=cfg_scale,
    )

    if not output_path.exists():
        raise RuntimeError(
            f"Keyframe pipeline produced no output at {output_path}"
        )

    logger.info(
        "video_keyframe_complete",
        path=str(output_path),
        fps=frame_rate,
        frames=num_frames,
    )
    return output_path


def _run_two_stage_pipeline(
    *,
    prompt: str,
    output_path: Path,
    model_dir: str,
    image: str | None,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    seed: int,
    stage1_steps: int | None,
    stage2_steps: int | None,
    cfg_scale: float,
    low_memory: bool,
) -> Path:
    """Two-stage T2V/I2V: dev model + CFG + upscale."""
    from ltx_pipelines_mlx import TI2VidTwoStagesPipeline

    logger.info(
        "video_two_stage_start",
        model=model_dir,
        image=image,
        frames=num_frames,
    )

    pipe = TI2VidTwoStagesPipeline(
        model_dir=model_dir,
        low_memory=low_memory,
    )

    pipe.generate_and_save(
        prompt=prompt,
        output_path=str(output_path),
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        seed=seed,
        image=image,
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        cfg_scale=cfg_scale,
    )

    if not output_path.exists():
        raise RuntimeError(
            f"Two-stage pipeline produced no output at {output_path}"
        )

    logger.info("video_two_stage_complete", path=str(output_path))
    return output_path


def _run_hq_pipeline(
    *,
    prompt: str,
    output_path: Path,
    model_dir: str,
    image: str | None,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    seed: int,
    stage1_steps: int | None,
    stage2_steps: int | None,
    cfg_scale: float,
    low_memory: bool,
) -> Path:
    """HQ pipeline: res_2s sampler + CFG + upscale."""
    from ltx_pipelines_mlx import TI2VidTwoStagesHQPipeline

    logger.info("video_hq_start", model=model_dir, frames=num_frames)

    pipe = TI2VidTwoStagesHQPipeline(
        model_dir=model_dir,
        low_memory=low_memory,
    )

    pipe.generate_and_save(
        prompt=prompt,
        output_path=str(output_path),
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        seed=seed,
        image=image,
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        cfg_scale=cfg_scale,
    )

    if not output_path.exists():
        raise RuntimeError(f"HQ pipeline produced no output at {output_path}")

    logger.info("video_hq_complete", path=str(output_path))
    return output_path


def _run_distilled_pipeline(
    *,
    prompt: str,
    output_path: Path,
    model_dir: str,
    image: str | None,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    seed: int,
    stage1_steps: int | None,
    stage2_steps: int | None,
    low_memory: bool,
) -> Path:
    """Distilled: fastest mode, half-res + upscale."""
    from ltx_pipelines_mlx import DistilledPipeline

    logger.info("video_distilled_start", model=model_dir, frames=num_frames)

    pipe = DistilledPipeline(
        model_dir=model_dir,
        low_memory=low_memory,
    )

    pipe.generate_and_save(
        prompt=prompt,
        output_path=str(output_path),
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        seed=seed,
        image=image,
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
    )

    if not output_path.exists():
        raise RuntimeError(
            f"Distilled pipeline produced no output at {output_path}"
        )

    logger.info("video_distilled_complete", path=str(output_path))
    return output_path


def _run_one_stage_pipeline(
    *,
    prompt: str,
    output_path: Path,
    model_dir: str,
    image: str | None,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    seed: int,
    cfg_scale: float,
    low_memory: bool,
) -> Path:
    """One-stage: full-res CFG, no upscaler."""
    from ltx_pipelines_mlx import TI2VidOneStagePipeline

    logger.info("video_one_stage_start", model=model_dir, frames=num_frames)

    pipe = TI2VidOneStagePipeline(
        model_dir=model_dir,
        low_memory=low_memory,
    )

    pipe.generate_and_save(
        prompt=prompt,
        output_path=str(output_path),
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        seed=seed,
        image=image,
        cfg_scale=cfg_scale,
    )

    if not output_path.exists():
        raise RuntimeError(
            f"One-stage pipeline produced no output at {output_path}"
        )

    logger.info("video_one_stage_complete", path=str(output_path))
    return output_path


def _run_ic_lora_pipeline(
    *,
    prompt: str,
    output_path: Path,
    model_dir: str,
    image: str | None,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    seed: int,
    stage1_steps: int | None,
    stage2_steps: int | None,
    low_memory: bool,
    lora_paths: list[tuple[str, float]],
    video_conditioning: list[tuple[str, float]],
    conditioning_strength: float,
    skip_stage_2: bool,
) -> Path:
    """IC-LoRA: conditioned generation with reference video + LoRA adapters.

    Uses distilled model (no CFG). Two-stage with optional upscale.
    """
    from ltx_pipelines_mlx import ICLoraPipeline

    logger.info(
        "video_ic_lora_start",
        model=model_dir,
        lora_paths=lora_paths,
        video_conditioning=video_conditioning,
        frames=num_frames,
    )

    pipe = ICLoraPipeline(
        model_dir=model_dir,
        low_memory=low_memory,
    )

    pipe.generate_and_save(
        prompt=prompt,
        output_path=str(output_path),
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        seed=seed,
        image=image,
        lora_paths=lora_paths,
        video_conditioning=video_conditioning,
        conditioning_strength=conditioning_strength,
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        skip_stage_2=skip_stage_2,
    )

    if not output_path.exists():
        raise RuntimeError(
            f"IC-LoRA pipeline produced no output at {output_path}"
        )

    logger.info("video_ic_lora_complete", path=str(output_path))
    return output_path


def _run_hdr_ic_lora_pipeline(
    *,
    prompt: str,
    output_path: Path,
    model_dir: str,
    image: str | None,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    seed: int,
    stage1_steps: int | None,
    stage2_steps: int | None,
    low_memory: bool,
    lora_paths: list[tuple[str, float]],
    video_conditioning: list[tuple[str, float]],
    conditioning_strength: float,
    skip_stage_2: bool,
) -> Path:
    """HDR IC-LoRA: SDR mp4 + HDR .npz output.

    Same as IC-LoRA but produces HDR output via HDRICLoraPipeline.
    Video conditioning is optional for HDR mode.
    """
    from ltx_pipelines_mlx import HDRICLoraPipeline

    logger.info(
        "video_hdr_ic_lora_start",
        model=model_dir,
        lora_paths=lora_paths,
        frames=num_frames,
    )

    pipe = HDRICLoraPipeline(
        model_dir=model_dir,
        low_memory=low_memory,
    )

    pipe.generate_and_save(
        prompt=prompt,
        output_path=str(output_path),
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        seed=seed,
        image=image,
        lora_paths=lora_paths,
        video_conditioning=video_conditioning,
        conditioning_strength=conditioning_strength,
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        skip_stage_2=skip_stage_2,
    )

    if not output_path.exists():
        raise RuntimeError(
            f"HDR IC-LoRA pipeline produced no output at {output_path}"
        )

    logger.info("video_hdr_ic_lora_complete", path=str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Convenience functions — video
# ---------------------------------------------------------------------------


def generate_video(
    prompt: str,
    output: str | Path,
    *,
    mode: VideoMode | str = "two-stage",
    image: str | Path | None = None,
    start_image: str | Path | None = None,
    end_image: str | Path | None = None,
    duration: float | None = None,
    fps: int = 24,
    width: int = 704,
    height: int = 480,
    num_frames: int | None = None,
    seed: int = 42,
    cfg_scale: float = 3.0,
    model_dir: str | None = None,
    lora_paths: list[LoRAAdapter | tuple[str, float]] | None = None,
    video_conditioning: list[tuple[str, float]] | None = None,
    conditioning_strength: float = 1.0,
) -> Path:
    """Convenience function for video generation.

    Supports all LTX-2.3 modes including keyframe interpolation and IC-LoRA.
    When both start_image and end_image are provided, automatically
    uses keyframe interpolation mode. When lora_paths and video_conditioning
    are provided, automatically uses ic-lora mode.
    """
    gen = VideoGenerator(model_dir=model_dir, seed=seed, fps=fps)
    return gen.generate(
        prompt,
        output,
        mode=mode,
        image=image,
        start_image=start_image,
        end_image=end_image,
        duration=duration,
        width=width,
        height=height,
        num_frames=num_frames,
        cfg_scale=cfg_scale,
        lora_paths=lora_paths,
        video_conditioning=video_conditioning,
        conditioning_strength=conditioning_strength,
    )
