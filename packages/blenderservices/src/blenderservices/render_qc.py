"""Phase 7 render QC and chapter assembly helpers."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

# TODO: import from scriptforge
# from forge.domain.artifact import VideoClip
from blenderservices.episode_settings import load_chapter_render_settings
from blenderservices.scaffold import load_shot_list_for_chapter, shots_dir
from blenderservices.shot_list import EpisodeShotPlan, ShotListDocument


# TODO: import from scriptforge — placeholders until scriptforge provides these
class VideoClip:
    """Placeholder until scriptforge provides forge.domain.artifact.VideoClip."""

    def __init__(self, path: Any, duration_s: float = 0.0, fps: float = 24.0):
        self.path = path
        self.duration_s = duration_s
        self.fps = fps


# TODO: import from scriptforge
# from forge.video_gen.assemble import AssemblyValidationError, concat_clips, mux_audio_video
# from forge.video_gen.probe import probe_media
# from forge.video_gen.validate import validate_clip_timing, validate_clips


def concat_clips(clips: list[VideoClip], output: Path, *, fps: int = 24) -> None:
    """Placeholder until scriptforge provides forge.video_gen.assemble.concat_clips."""
    raise NotImplementedError("TODO: import concat_clips from scriptforge")


def mux_audio_video(video: Path, audio: Path, output: Path, **kwargs: Any) -> None:
    """Placeholder until scriptforge provides forge.video_gen.assemble.mux_audio_video."""
    raise NotImplementedError("TODO: import mux_audio_video from scriptforge")


class AssemblyValidationError(Exception):
    """Placeholder until scriptforge provides forge.video_gen.assemble.AssemblyValidationError."""

    pass


def probe_media(path: Any) -> Any:
    """Placeholder until scriptforge provides forge.video_gen.probe.probe_media."""
    raise NotImplementedError("TODO: import probe_media from scriptforge")


def validate_clip_timing(clip: Any, **kwargs: Any) -> list[str]:
    """Placeholder until scriptforge provides forge.video_gen.validate.validate_clip_timing."""
    return []


def validate_clips(clips: list[VideoClip], **kwargs: Any) -> list[str]:
    """Placeholder until scriptforge provides forge.video_gen.validate.validate_clips."""
    return []


class QCStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ShotTimingReport:
    shot_id: str
    planned_duration_s: float
    actual_duration_s: float | None
    planned_fps: int
    actual_fps: float | None
    issues: tuple[str, ...]


@dataclass(frozen=True)
class ChapterTimingReport:
    chapter: int
    episode: str
    planned_total_s: float
    actual_total_s: float
    missing_shots: tuple[str, ...]
    shots: tuple[ShotTimingReport, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_shots and all(not shot.issues for shot in self.shots)


@dataclass(frozen=True)
class QCFrameSpec:
    shot_id: str
    frame: int
    label: str


@dataclass(frozen=True)
class ShotQCResult:
    shot_id: str
    blend_path: Path
    status: QCStatus
    frames: tuple[QCFrameSpec, ...]
    output_dir: Path | None
    message: str


def qc_frame_numbers(frame_start: int, frame_end: int) -> tuple[int, ...]:
    """Return start, middle, and end frames for visual QC."""
    if frame_end < frame_start:
        return (frame_start,)
    if frame_start == frame_end:
        return (frame_start,)
    mid = frame_start + (frame_end - frame_start) // 2
    return (frame_start, mid, frame_end)


def qc_frame_specs(shot_id: str, frame_start: int, frame_end: int) -> tuple[QCFrameSpec, ...]:
    labels = ("start", "mid", "end")
    frames = qc_frame_numbers(frame_start, frame_end)
    if len(frames) == 1:
        return (QCFrameSpec(shot_id, frames[0], "start"),)
    if len(frames) == 2:
        return (
            QCFrameSpec(shot_id, frames[0], "start"),
            QCFrameSpec(shot_id, frames[1], "end"),
        )
    return tuple(
        QCFrameSpec(shot_id, frame, label) for frame, label in zip(frames, labels, strict=True)
    )


def _resolve_shot_blend(project_root: Path, doc: ShotListDocument, shot_id: str) -> Path:
    shots = shots_dir(project_root, doc)
    suffix = shot_id.rsplit("_", maxsplit=1)[-1]
    candidate = shots / f"{suffix}.blend"
    if candidate.is_file():
        return candidate
    msg = f"missing blend for {shot_id}: {candidate}"
    raise FileNotFoundError(msg)


def render_qc_output_dir(project_root: Path, doc: ShotListDocument, shot_id: str) -> Path:
    return project_root / "build" / "renders" / "qc" / doc.episode / doc.reel / shot_id


def run_blender_render_qc(
    blend_path: Path,
    output_dir: Path,
    frames: tuple[int, ...],
    *,
    blender: str | None = None,
) -> None:
    script = Path(__file__).resolve().parents[3] / "tools" / "render_qc.py"
    if not script.is_file():
        raise RuntimeError(f"missing Blender QC script: {script}")

    blender_bin = blender or "blender"
    cmd = [
        blender_bin,
        "--background",
        str(blend_path.resolve()),
        "--python",
        str(script),
        "--",
        "--output-dir",
        str(output_dir.resolve()),
        "--frames",
        ",".join(str(frame) for frame in frames),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip()
        raise RuntimeError(f"render QC failed for {blend_path.name}: {detail}")


def render_shot_qc(
    project_root: Path,
    doc: ShotListDocument,
    shot_id: str,
    *,
    blender: str | None = None,
) -> ShotQCResult:
    blend_path = _resolve_shot_blend(project_root, doc, shot_id)
    plan = next((shot for shot in doc.shots if shot.id == shot_id), None)
    frame_end = plan.frames if plan is not None else 240
    specs = qc_frame_specs(shot_id, 1, frame_end)
    frames = tuple(spec.frame for spec in specs)
    output_dir = render_qc_output_dir(project_root, doc, shot_id)

    try:
        run_blender_render_qc(blend_path, output_dir, frames, blender=blender)
    except RuntimeError as exc:
        return ShotQCResult(
            shot_id=shot_id,
            blend_path=blend_path,
            status=QCStatus.FAIL,
            frames=specs,
            output_dir=None,
            message=str(exc),
        )

    missing = [
        spec for spec in specs if not (output_dir / f"{spec.label}_f{spec.frame:04d}.png").is_file()
    ]
    if missing:
        labels = ", ".join(spec.label for spec in missing)
        return ShotQCResult(
            shot_id=shot_id,
            blend_path=blend_path,
            status=QCStatus.FAIL,
            frames=specs,
            output_dir=output_dir,
            message=f"missing QC PNGs: {labels}",
        )

    manifest_path = output_dir / "qc_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "shot_id": shot_id,
                "blend_path": str(blend_path),
                "frames": [{"label": spec.label, "frame": spec.frame} for spec in specs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ShotQCResult(
        shot_id=shot_id,
        blend_path=blend_path,
        status=QCStatus.PASS,
        frames=specs,
        output_dir=output_dir,
        message=f"rendered {len(specs)} QC still(s)",
    )


def render_chapter_qc(
    project_root: Path,
    chapter_num: int,
    *,
    blender: str | None = None,
    only_shots: set[str] | None = None,
    require_characters: set[str] | None = None,
) -> list[ShotQCResult]:
    doc = load_shot_list_for_chapter(project_root, chapter_num)
    results: list[ShotQCResult] = []
    for plan in doc.shots:
        if only_shots is not None and plan.id not in only_shots:
            continue
        if require_characters is not None:
            if not require_characters.intersection(plan.character_ids):
                continue
        results.append(render_shot_qc(project_root, doc, plan.id, blender=blender))
    return results


def default_chapter_narration_path(project_root: Path, chapter_num: int) -> Path:
    doc = load_shot_list_for_chapter(project_root, chapter_num)
    return (
        project_root
        / "build"
        / "renders"
        / doc.episode
        / "narration"
        / f"chapter_{chapter_num:02d}.wav"
    )


def run_blender_pilot_render(
    blend_path: Path,
    output_mp4: Path,
    *,
    frame_end: int = 24,
    blender: str | None = None,
) -> None:
    run_blender_shot_render(blend_path, output_mp4, frame_end=frame_end, blender=blender)


def _run_blender_logged(cmd: list[str], *, fail_label: str) -> None:
    """Run a Blender subprocess, streaming its (very chatty) stdout/stderr to a
    temp file rather than a pipe.

    A full-length render floods stdout with per-frame + per-compositing-operation
    lines. `capture_output=True` only drains the pipe after the process exits, so
    once the 64 KB OS pipe buffer fills, Blender blocks on write and the render
    deadlocks. Writing to a file keeps the buffer from ever filling.
    """
    with tempfile.NamedTemporaryFile("w+", suffix=".log", delete=True) as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            log.seek(0)
            detail = log.read().strip()[-2000:]
            raise RuntimeError(f"{fail_label}: {detail}")


def run_blender_shot_render(
    blend_path: Path,
    output_mp4: Path,
    *,
    frame_end: int | None = None,
    fps: int = 24,
    blender: str | None = None,
) -> None:
    script = Path(__file__).resolve().parents[3] / "tools" / "render_shot_mp4.py"
    blender_bin = blender or "blender"
    cmd = [
        blender_bin,
        "--background",
        str(blend_path.resolve()),
        "--python",
        str(script),
        "--",
        "--output",
        str(output_mp4.resolve()),
        "--fps",
        str(fps),
    ]
    if frame_end is not None:
        cmd += ["--frame-end", str(frame_end)]
    _run_blender_logged(cmd, fail_label=f"shot render failed for {blend_path.name}")


@dataclass(frozen=True)
class ShotRenderResult:
    shot_id: str
    output: Path | None
    message: str


#: EEVEE TAA sample counts. Exteriors have no characters but ~60 dense foliage
#: meshes + the heavy compositor grade, so each frame is ~15x slower than an
#: interior; render them at fewer samples (still clean enough for a wide that
#: holds no facial detail) to keep the chapter render tractable on one machine.
INTERIOR_SAMPLES = 56
EXTERIOR_SAMPLES = 24


@dataclass(frozen=True)
class RenderProfile:
    """Per-shot look-dev render settings, derived from the shot plan."""

    no_freestyle: bool
    relight: str
    faces: bool
    samples: int = INTERIOR_SAMPLES
    mood: str = "auto"


def render_profile_for(plan: EpisodeShotPlan) -> RenderProfile:
    """Map plan fields to a look-dev profile.

    - Exteriors disable Freestyle (its view-map on dense foliage stalls render for hours)
      and render at a lower sample count (no characters, heavy foliage = slow frames).
    - Hut interiors get the carved-wood chiaroscuro relight.
    - Hero (close-up) cameras get Hunyuan face meshes; wides/mediums skip them.
    """
    set_id = (plan.set_id or "").lower()
    camera = (plan.camera or "").lower()
    is_exterior = set_id.endswith("exterior")
    # relight defaults off: the chiaroscuro rig is tuned for the shot07 hero camera and
    # crushes other shots to black. Generic shots use their preset lighting + an exposure
    # lift (in render_shot_production) so carved-wood surfaces read. Per-shot chiaroscuro
    # is a later tuning pass.
    return RenderProfile(
        no_freestyle=is_exterior,
        relight="none",
        faces=camera == "cam_hero",
        samples=EXTERIOR_SAMPLES if is_exterior else INTERIOR_SAMPLES,
    )


def run_blender_production_render(
    blend_path: Path,
    output_mp4: Path,
    profile: RenderProfile,
    *,
    frame_end: int | None = None,
    fps: int = 24,
    frame_step: int = 3,
    blender: str | None = None,
) -> None:
    script = Path(__file__).resolve().parents[3] / "tools" / "render_shot_production.py"
    blender_bin = blender or "blender"
    cmd = [
        blender_bin,
        "--background",
        str(blend_path.resolve()),
        "--python",
        str(script),
        "--",
        "--output",
        str(output_mp4.resolve()),
        "--fps",
        str(fps),
        "--frame-step",
        str(frame_step),
        "--mood",
        profile.mood,
        "--relight",
        profile.relight,
        "--samples",
        str(profile.samples),
    ]
    if profile.no_freestyle:
        cmd.append("--no-freestyle")
    if profile.faces:
        cmd.append("--faces")
    if frame_end is not None:
        cmd += ["--frame-end", str(frame_end)]
    _run_blender_logged(cmd, fail_label=f"production render failed for {blend_path.name}")


def render_shot_mp4(
    project_root: Path,
    doc: ShotListDocument,
    shot_id: str,
    *,
    frame_end: int | None = None,
    blender: str | None = None,
    lookdev: bool = True,
    frame_step: int = 3,
    plan: EpisodeShotPlan | None = None,
) -> ShotRenderResult:
    suffix = shot_id.rsplit("_", maxsplit=1)[-1]
    blend_path = shots_dir(project_root, doc) / f"{suffix}.blend"
    if not blend_path.is_file():
        return ShotRenderResult(shot_id, None, f"missing blend: {blend_path}")

    settings = load_chapter_render_settings(project_root, doc.chapter)
    output_mp4 = (
        project_root / "build" / "renders" / doc.episode / doc.reel / suffix / f"{suffix}.mp4"
    )
    try:
        if lookdev:
            if plan is None:
                plan = next((p for p in doc.shots if p.id == shot_id), None)
            if plan is None:
                return ShotRenderResult(shot_id, None, f"no plan for {shot_id}")
            run_blender_production_render(
                blend_path,
                output_mp4,
                render_profile_for(plan),
                frame_end=frame_end,
                fps=settings.fps,
                frame_step=frame_step,
                blender=blender,
            )
        else:
            run_blender_shot_render(
                blend_path,
                output_mp4,
                frame_end=frame_end,
                fps=settings.fps,
                blender=blender,
            )
    except RuntimeError as exc:
        return ShotRenderResult(shot_id, None, str(exc))
    return ShotRenderResult(shot_id, output_mp4, f"rendered {output_mp4.name}")


def render_chapter_mp4s(
    project_root: Path,
    chapter_num: int,
    *,
    frame_end: int | None = None,
    blender: str | None = None,
    only_shots: set[str] | None = None,
    lookdev: bool = True,
    frame_step: int = 3,
) -> list[ShotRenderResult]:
    doc = load_shot_list_for_chapter(project_root, chapter_num)
    results: list[ShotRenderResult] = []
    for plan in doc.shots:
        if only_shots is not None and plan.id not in only_shots:
            continue
        results.append(
            render_shot_mp4(
                project_root,
                doc,
                plan.id,
                frame_end=frame_end,
                blender=blender,
                lookdev=lookdev,
                frame_step=frame_step,
                plan=plan,
            )
        )
    return results


def shot_video_candidates(project_root: Path, doc: ShotListDocument, shot_id: str) -> list[Path]:
    suffix = shot_id.rsplit("_", maxsplit=1)[-1]
    roots = (
        project_root / "build" / "renders" / doc.episode / doc.reel / suffix,
        project_root / "build" / "renders" / suffix,
        project_root / "build" / "renders" / "mp4" / doc.episode / doc.reel,
        shots_dir(project_root, doc),
    )
    names = (f"{suffix}.mp4", f"{shot_id}.mp4")
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for name in names:
            path = root / name
            if path.is_file():
                found.append(path)
        found.extend(sorted(root.glob("*.mp4")))
    # preserve order, dedupe
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(path)
    return ordered


def resolve_shot_clip(
    project_root: Path,
    doc: ShotListDocument,
    shot_id: str,
    *,
    default_fps: int | None = None,
) -> VideoClip | None:
    candidates = shot_video_candidates(project_root, doc, shot_id)
    if not candidates:
        return None
    plan = next((shot for shot in doc.shots if shot.id == shot_id), None)
    duration = float(plan.duration_sec) if plan is not None else 0.0
    fps = default_fps
    if fps is None:
        fps = load_chapter_render_settings(project_root, doc.chapter).fps
    return VideoClip(path=candidates[0], duration_s=duration, fps=fps)


def chapter_timing_report(
    project_root: Path,
    chapter_num: int,
    *,
    tolerance_s: float = 0.1,
) -> ChapterTimingReport:
    doc = load_shot_list_for_chapter(project_root, chapter_num)
    settings = load_chapter_render_settings(project_root, chapter_num)
    shots: list[ShotTimingReport] = []
    missing: list[str] = []
    actual_total = 0.0
    planned_total = float(sum(plan.duration_sec for plan in doc.shots))

    for plan in doc.shots:
        clip = resolve_shot_clip(project_root, doc, plan.id, default_fps=settings.fps)
        if clip is None:
            missing.append(plan.id)
            shots.append(
                ShotTimingReport(
                    shot_id=plan.id,
                    planned_duration_s=float(plan.duration_sec),
                    actual_duration_s=None,
                    planned_fps=settings.fps,
                    actual_fps=None,
                    issues=(f"missing MP4 for {plan.id}",),
                )
            )
            continue

        issues = tuple(validate_clip_timing(clip, tolerance_s=tolerance_s, fps_tolerance=0.01))
        try:
            probe = probe_media(clip.path)
            actual_total += probe.duration_s
            shots.append(
                ShotTimingReport(
                    shot_id=plan.id,
                    planned_duration_s=float(plan.duration_sec),
                    actual_duration_s=probe.duration_s,
                    planned_fps=settings.fps,
                    actual_fps=probe.fps,
                    issues=issues,
                )
            )
        except RuntimeError as exc:
            shots.append(
                ShotTimingReport(
                    shot_id=plan.id,
                    planned_duration_s=float(plan.duration_sec),
                    actual_duration_s=None,
                    planned_fps=settings.fps,
                    actual_fps=None,
                    issues=(str(exc),),
                )
            )

    return ChapterTimingReport(
        chapter=chapter_num,
        episode=doc.episode,
        planned_total_s=planned_total,
        actual_total_s=actual_total,
        missing_shots=tuple(missing),
        shots=tuple(shots),
    )


def assemble_chapter_from_renders(
    project_root: Path,
    chapter_num: int,
    output_path: Path,
    narration_path: Path | None = None,
    *,
    skip_validation: bool = False,
    tolerance_s: float = 0.1,
) -> tuple[Path, list[str]]:
    """Concatenate shot MP4s in shot-list order. Returns output path and missing shot ids."""
    doc = load_shot_list_for_chapter(project_root, chapter_num)
    settings = load_chapter_render_settings(project_root, chapter_num)
    clips: list[VideoClip] = []
    missing: list[str] = []
    for plan in doc.shots:
        clip = resolve_shot_clip(project_root, doc, plan.id, default_fps=settings.fps)
        if clip is None:
            missing.append(plan.id)
            continue
        clips.append(clip)

    if not clips:
        msg = "no shot MP4s found under build/renders/"
        raise FileNotFoundError(msg)

    if missing:
        msg = f"missing {len(missing)} shot MP4(s): {', '.join(missing)}"
        raise FileNotFoundError(msg)

    if not skip_validation:
        issues = validate_clips(clips, tolerance_s=tolerance_s)
        if issues:
            detail = "\n".join(f"  - {issue}" for issue in issues)
            msg = f"clip validation failed:\n{detail}"
            raise AssemblyValidationError(msg)

    target_duration_s = sum(c.duration_s for c in clips)

    if narration_path is not None and narration_path.is_file():
        silent = output_path.with_name(output_path.stem + "_silent.mp4")
        concat_clips(clips, silent, fps=settings.fps)
        try:
            mux_audio_video(
                silent,
                narration_path,
                output_path,
                target_duration_s=target_duration_s,
                tolerance_s=max(tolerance_s, 0.5),
            )
        except AssemblyValidationError:
            silent.unlink(missing_ok=True)
            raise
        silent.unlink(missing_ok=True)
    else:
        concat_clips(clips, output_path, fps=settings.fps)

    return output_path, missing
