"""Apply manifest assets, camera, and lighting to scaffolded shots."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blenderservices.shot_animation import AnimateResult

from blenderservices.scaffold import load_shot_list_for_chapter, shots_dir
from blenderservices.shot_animation import AnimateStatus, animate_shot
from blenderservices.shot_list import EpisodeShotPlan, ShotListDocument


class ApplyStatus(StrEnum):
    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class ApplyResult:
    shot_id: str
    status: ApplyStatus
    message: str


def _resolve_shot_blend(project_root: Path, doc: ShotListDocument, shot_id: str) -> Path:
    suffix = shot_id.rsplit("_", maxsplit=1)[-1]
    candidate = shots_dir(project_root, doc) / f"{suffix}.blend"
    if not candidate.is_file():
        msg = f"missing blend for {shot_id}: {candidate}"
        raise FileNotFoundError(msg)
    return candidate


def _run_blender_script(
    blend_path: Path,
    script: Path,
    script_args: list[str],
    *,
    blender: str | None = None,
) -> subprocess.CompletedProcess[str]:
    blender_bin = blender or "blender"
    cmd = [
        blender_bin,
        "--background",
        str(blend_path.resolve()),
        "--python",
        str(script),
        "--",
        *script_args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def link_shot_assets(
    blend_path: Path,
    plan: EpisodeShotPlan,
    repo_root: Path,
    *,
    blender: str | None = None,
    replace: bool = True,
    character_source: str = "approved",
) -> ApplyResult:
    script = repo_root / "tools" / "link_shot_assets.py"
    args = ["--replace"] if replace else []
    args += ["--character-source", character_source]
    if plan.set_id:
        args += ["--set", plan.set_id]
    if plan.character_ids:
        args += ["--characters", ",".join(plan.character_ids)]
    if plan.prop_ids:
        args += ["--props", ",".join(plan.prop_ids)]
    if plan.action_library_ids:
        args += ["--actions", ",".join(plan.action_library_ids)]

    result = _run_blender_script(blend_path, script, args, blender=blender)
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip()
        return ApplyResult(plan.id, ApplyStatus.FAILED, f"link failed: {detail}")

    parts = [plan.set_id or "—"]
    if plan.character_ids:
        parts.append(",".join(plan.character_ids))
    if plan.prop_ids:
        parts.append(f"props:{','.join(plan.prop_ids)}")
    return ApplyResult(plan.id, ApplyStatus.APPLIED, " · ".join(parts))


def apply_shot_setup(
    blend_path: Path,
    plan: EpisodeShotPlan,
    repo_root: Path,
    *,
    blender: str | None = None,
) -> ApplyResult:
    if not plan.camera or not plan.lighting:
        return ApplyResult(plan.id, ApplyStatus.SKIPPED, "no camera/lighting in plan")

    script = repo_root / "tools" / "apply_shot_setup.py"
    result = _run_blender_script(
        blend_path,
        script,
        ["--camera", plan.camera, "--lighting", plan.lighting],
        blender=blender,
    )
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip()
        return ApplyResult(plan.id, ApplyStatus.FAILED, detail)
    return ApplyResult(plan.id, ApplyStatus.APPLIED, f"{plan.camera} · {plan.lighting}")


def apply_story_actions(
    blend_path: Path,
    plan: EpisodeShotPlan,
    repo_root: Path,
    *,
    blender: str | None = None,
    quality: str = "daily",
) -> AnimateResult:
    from blenderservices.shot_animation import AnimateResult  # local to avoid API churn

    if not plan.character_actions:
        return AnimateResult(plan.id, AnimateStatus.SKIPPED, "no character_actions")

    script = repo_root / "tools" / "apply_story_actions.py"
    actions_arg = ",".join(
        f"{char}:{action}" for char, action in sorted(plan.character_actions.items())
    )
    result = _run_blender_script(
        blend_path,
        script,
        ["--actions", actions_arg, "--quality", quality],
        blender=blender,
    )
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip()
        return AnimateResult(plan.id, AnimateStatus.FAILED, detail)
    summary = ", ".join(f"{k}:{v}" for k, v in plan.character_actions.items())
    return AnimateResult(plan.id, AnimateStatus.APPLIED, f"story:{summary}")


def stage_shot_layout(
    blend_path: Path,
    plan: EpisodeShotPlan,
    repo_root: Path,
    *,
    blender: str | None = None,
) -> ApplyResult:
    script = repo_root / "tools" / "stage_shot_layout.py"
    result = _run_blender_script(
        blend_path,
        script,
        ["--shot", plan.id],
        blender=blender,
    )
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip()
        return ApplyResult(plan.id, ApplyStatus.FAILED, f"stage failed: {detail}")
    return ApplyResult(plan.id, ApplyStatus.APPLIED, "layout")


def apply_shot_from_plan(
    blend_path: Path,
    plan: EpisodeShotPlan,
    repo_root: Path,
    *,
    blender: str | None = None,
    replace_assets: bool = True,
    character_source: str = "approved",
    action_source: str = "procedural",
) -> ApplyResult:
    """Relink catalog assets, then align camera and lighting from the shot plan."""
    link = link_shot_assets(
        blend_path,
        plan,
        repo_root,
        blender=blender,
        replace=replace_assets,
        character_source=character_source,
    )
    if link.status == ApplyStatus.FAILED:
        return link

    setup = apply_shot_setup(blend_path, plan, repo_root, blender=blender)
    if setup.status == ApplyStatus.FAILED:
        return setup

    parts = [link.message, setup.message]
    if plan.character_actions and replace_assets:
        if action_source == "story":
            anim = apply_story_actions(blend_path, plan, repo_root, blender=blender, quality="hero")
        else:
            anim = animate_shot(blend_path, plan, repo_root, blender=blender, quality="hero")
        if anim.status == AnimateStatus.FAILED:
            return ApplyResult(plan.id, ApplyStatus.FAILED, anim.message)
        parts.append(anim.message)

    stage = stage_shot_layout(blend_path, plan, repo_root, blender=blender)
    if stage.status == ApplyStatus.FAILED:
        return ApplyResult(plan.id, ApplyStatus.FAILED, stage.message)
    parts.append(stage.message)

    # Staging repositions cam_* anchors; refresh cam_master from the plan camera.
    if plan.camera and plan.lighting:
        setup2 = apply_shot_setup(blend_path, plan, repo_root, blender=blender)
        if setup2.status == ApplyStatus.FAILED:
            return ApplyResult(plan.id, ApplyStatus.FAILED, setup2.message)

    return ApplyResult(
        plan.id,
        ApplyStatus.APPLIED,
        " · ".join(parts),
    )


def apply_chapter_shots(
    project_root: Path,
    chapter_num: int,
    *,
    blender: str | None = None,
    only_shots: set[str] | None = None,
    require_characters: set[str] | None = None,
    min_shot_number: int | None = None,
    replace_assets: bool = True,
    character_source: str = "approved",
    action_source: str = "procedural",
) -> list[ApplyResult]:
    project_root = project_root.resolve()
    repo_root = project_root
    doc = load_shot_list_for_chapter(project_root, chapter_num)
    results: list[ApplyResult] = []

    for plan in doc.shots:
        if only_shots is not None and plan.id not in only_shots:
            continue
        if min_shot_number is not None:
            shot_num = int(plan.id.rsplit("_", maxsplit=1)[-1].removeprefix("shot"))
            if shot_num < min_shot_number:
                continue
        if require_characters is not None:
            if not require_characters.intersection(plan.character_ids):
                continue
        try:
            blend_path = _resolve_shot_blend(project_root, doc, plan.id)
        except FileNotFoundError as exc:
            results.append(ApplyResult(plan.id, ApplyStatus.FAILED, str(exc)))
            continue
        results.append(
            apply_shot_from_plan(
                blend_path,
                plan,
                repo_root,
                blender=blender,
                replace_assets=replace_assets,
                character_source=character_source,
                action_source=action_source,
            )
        )

    return results
