"""Scaffold episode shot .blend files from a ShotListDocument."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from blenderservices.episode_settings import (
    EpisodeRenderSettings,
    episode_dir_for_doc,
    load_episode_render_settings,
)
from blenderservices.shot_list import EpisodeShotPlan, ShotListDocument, load_shot_list


class ScaffoldStatus(StrEnum):
    CREATED = "created"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class ScaffoldResult:
    shot_id: str
    blend_path: Path | None
    status: ScaffoldStatus
    message: str


def shots_dir(project_root: Path, doc: ShotListDocument) -> Path:
    return project_root / "shot_production" / doc.episode / "reels" / doc.reel / "shots"


def shot_number(shot_id: str) -> str:
    match = re.search(r"shot(\d{2})$", shot_id)
    if match is None:
        msg = f"invalid shot id: {shot_id}"
        raise ValueError(msg)
    return match.group(1)


def scene_name(shot_id: str) -> str:
    """ep01_reel01_shot02 → same (already the scene naming convention)."""
    return shot_id


def _plan_has_assets(plan: EpisodeShotPlan) -> bool:
    return bool(plan.set_id or plan.character_ids or plan.prop_ids or plan.action_library_ids)


def _patch_shot_blend(
    target: Path,
    plan: EpisodeShotPlan,
    repo_root: Path,
    settings: EpisodeRenderSettings,
    *,
    blender: str | None = None,
) -> subprocess.CompletedProcess[str]:
    blender_bin = blender or shutil.which("blender") or "blender"
    patch_script = repo_root / "tools" / "_patch_shot.py"
    patch_cmd = [
        blender_bin,
        "--background",
        "--factory-startup",
        str(target),
        "--python",
        str(patch_script),
        "--",
        "--scene-name",
        scene_name(plan.id),
        "--output",
        f"//../../../../../build/renders/{scene_name(plan.id)}/####",
        "--frames",
        str(plan.frames),
        "--fps",
        str(settings.fps),
        "--resolution-x",
        str(settings.width),
        "--resolution-y",
        str(settings.height),
    ]
    return subprocess.run(patch_cmd, capture_output=True, text=True, check=False)


def scaffold_shot_from_plan(
    plan: EpisodeShotPlan,
    target_dir: Path,
    repo_root: Path,
    *,
    blender: str | None = None,
    skip_existing: bool = True,
    render_settings: EpisodeRenderSettings | None = None,
) -> ScaffoldResult:
    """Copy shot template, patch metadata, and link catalog assets for one plan."""
    num = shot_number(plan.id)
    blend_name = f"shot{num}.blend"
    target = target_dir / blend_name

    if skip_existing and target.exists():
        return ScaffoldResult(plan.id, target, ScaffoldStatus.SKIPPED, "blend already exists")

    template = repo_root / "templates" / "_shot.blend"
    if not template.is_file():
        return ScaffoldResult(
            plan.id,
            None,
            ScaffoldStatus.FAILED,
            f"template missing: {template}",
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template, target)

    settings = render_settings or EpisodeRenderSettings()
    patch = _patch_shot_blend(target, plan, repo_root, settings, blender=blender)
    if patch.returncode != 0:
        target.unlink(missing_ok=True)
        detail = (patch.stdout + patch.stderr).strip()
        return ScaffoldResult(plan.id, None, ScaffoldStatus.FAILED, f"patch failed: {detail}")

    blender_bin = blender or shutil.which("blender") or "blender"
    link_script = repo_root / "tools" / "link_shot_assets.py"

    if _plan_has_assets(plan):
        link_cmd = [blender_bin, "--background", str(target), "--python", str(link_script), "--"]
        if plan.set_id:
            link_cmd += ["--set", plan.set_id]
        if plan.character_ids:
            link_cmd += ["--characters", ",".join(plan.character_ids)]
        if plan.prop_ids:
            link_cmd += ["--props", ",".join(plan.prop_ids)]
        if plan.action_library_ids:
            link_cmd += ["--actions", ",".join(plan.action_library_ids)]

        link = subprocess.run(link_cmd, capture_output=True, text=True, check=False)
        if link.returncode != 0:
            target.unlink(missing_ok=True)
            detail = (link.stdout + link.stderr).strip()
            return ScaffoldResult(plan.id, None, ScaffoldStatus.FAILED, f"link failed: {detail}")

    if plan.camera and plan.lighting:
        setup_script = repo_root / "tools" / "apply_shot_setup.py"
        setup_cmd = [
            blender_bin,
            "--background",
            str(target),
            "--python",
            str(setup_script),
            "--",
            "--camera",
            plan.camera,
            "--lighting",
            plan.lighting,
        ]
        setup = subprocess.run(setup_cmd, capture_output=True, text=True, check=False)
        if setup.returncode != 0:
            target.unlink(missing_ok=True)
            detail = (setup.stdout + setup.stderr).strip()
            return ScaffoldResult(plan.id, None, ScaffoldStatus.FAILED, f"setup failed: {detail}")

    return ScaffoldResult(plan.id, target, ScaffoldStatus.CREATED, f"scaffolded {blend_name}")


def scaffold_from_shot_list(
    project_root: Path,
    doc: ShotListDocument,
    *,
    skip_existing: bool = True,
    only_shots: set[str] | None = None,
    blender: str | None = None,
) -> list[ScaffoldResult]:
    """Scaffold every shot in a shot list document."""
    repo_root = project_root.resolve()
    target_dir = shots_dir(repo_root, doc)
    settings = load_episode_render_settings(episode_dir_for_doc(repo_root, doc))
    results: list[ScaffoldResult] = []

    for plan in doc.shots:
        if only_shots is not None and plan.id not in only_shots:
            continue
        results.append(
            scaffold_shot_from_plan(
                plan,
                target_dir,
                repo_root,
                blender=blender,
                skip_existing=skip_existing,
                render_settings=settings,
            )
        )
    return results


def patch_shot_render_settings(
    plan: EpisodeShotPlan,
    blend_path: Path,
    repo_root: Path,
    settings: EpisodeRenderSettings,
    *,
    blender: str | None = None,
) -> ScaffoldResult:
    """Patch fps, resolution, and frame range on an existing shot blend."""
    if not blend_path.is_file():
        return ScaffoldResult(plan.id, None, ScaffoldStatus.FAILED, f"missing blend: {blend_path}")

    patch = _patch_shot_blend(blend_path, plan, repo_root, settings, blender=blender)
    if patch.returncode != 0:
        detail = (patch.stdout + patch.stderr).strip()
        return ScaffoldResult(plan.id, blend_path, ScaffoldStatus.FAILED, f"patch failed: {detail}")

    return ScaffoldResult(
        plan.id,
        blend_path,
        ScaffoldStatus.CREATED,
        f"patched {blend_path.name} ({settings.width}x{settings.height}@{settings.fps}fps)",
    )


def patch_chapter_shots(
    project_root: Path,
    chapter_num: int,
    *,
    only_shots: set[str] | None = None,
    blender: str | None = None,
) -> tuple[ShotListDocument, list[ScaffoldResult]]:
    """Apply episode render settings to existing chapter shot blends."""
    doc = load_shot_list_for_chapter(project_root, chapter_num)
    repo_root = project_root.resolve()
    settings = load_episode_render_settings(episode_dir_for_doc(repo_root, doc))
    target_dir = shots_dir(repo_root, doc)
    results: list[ScaffoldResult] = []

    for plan in doc.shots:
        if only_shots is not None and plan.id not in only_shots:
            continue
        num = shot_number(plan.id)
        blend_path = target_dir / f"shot{num}.blend"
        results.append(
            patch_shot_render_settings(
                plan,
                blend_path,
                repo_root,
                settings,
                blender=blender,
            )
        )
    return doc, results


def scaffold_chapter_shots(
    project_root: Path,
    chapter_num: int,
    *,
    skip_existing: bool = True,
    only_shots: set[str] | None = None,
    blender: str | None = None,
) -> tuple[ShotListDocument, list[ScaffoldResult]]:
    """Load episodes/epNN/shot_list.yaml and scaffold its shots."""
    doc = load_shot_list_for_chapter(project_root, chapter_num)
    results = scaffold_from_shot_list(
        project_root,
        doc,
        skip_existing=skip_existing,
        only_shots=only_shots,
        blender=blender,
    )
    return doc, results


def load_shot_list_for_chapter(project_root: Path, chapter_num: int) -> ShotListDocument:
    ch = f"{chapter_num:02d}"
    episodes = project_root / "shot_production"
    if not episodes.is_dir():
        msg = f"shot_production/ missing under {project_root}"
        raise FileNotFoundError(msg)

    matches = sorted(
        path for path in episodes.iterdir() if path.is_dir() and path.name.startswith(f"ep{ch}")
    )
    if not matches:
        msg = f"no episode folder for chapter {chapter_num}; run forge-3d shot-list --write first"
        raise FileNotFoundError(msg)

    shot_list_path = matches[0] / "shot_list.yaml"
    if not shot_list_path.is_file():
        msg = f"shot list missing: {shot_list_path}"
        raise FileNotFoundError(msg)
    return load_shot_list(shot_list_path)
