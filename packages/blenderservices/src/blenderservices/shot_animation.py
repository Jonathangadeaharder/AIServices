"""Apply procedural blockout animation from shot manifests."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from blenderservices.scaffold import load_shot_list_for_chapter, shots_dir
from blenderservices.shot_list import EpisodeShotPlan, ShotListDocument


class AnimateStatus(StrEnum):
    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class AnimateResult:
    shot_id: str
    status: AnimateStatus
    message: str


def _resolve_shot_blend(project_root: Path, doc: ShotListDocument, shot_id: str) -> Path:
    suffix = shot_id.rsplit("_", maxsplit=1)[-1]
    candidate = shots_dir(project_root, doc) / f"{suffix}.blend"
    if not candidate.is_file():
        msg = f"missing blend for {shot_id}: {candidate}"
        raise FileNotFoundError(msg)
    return candidate


def _travel_for_plan(plan: EpisodeShotPlan) -> float:
    if plan.set_id in {"forest_path", "forest_edge", "deep_forest"}:
        return 3.0
    if plan.set_id in {"gingerbread_house_exterior", "gingerbread_house_interior"}:
        return 0.8
    if plan.set_id == "root_shelter":
        return 0.0
    if plan.set_id == "hut_interior" and all(
        action == "idle" for action in plan.character_actions.values()
    ):
        return 0.0
    return 1.5


def animate_shot(
    blend_path: Path,
    plan: EpisodeShotPlan,
    repo_root: Path,
    *,
    blender: str | None = None,
    quality: str = "daily",
) -> AnimateResult:
    if not plan.character_actions:
        return AnimateResult(plan.id, AnimateStatus.SKIPPED, "no character_actions")

    script = repo_root / "tools" / "apply_shot_actions.py"
    actions_arg = ",".join(
        f"{char}:{action}" for char, action in sorted(plan.character_actions.items())
    )
    blender_bin = blender or "blender"
    cmd = [
        blender_bin,
        "--background",
        str(blend_path.resolve()),
        "--python",
        str(script),
        "--",
        "--actions",
        actions_arg,
        "--travel",
        str(_travel_for_plan(plan)),
        "--quality",
        quality,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip()
        return AnimateResult(plan.id, AnimateStatus.FAILED, detail)
    summary = ", ".join(f"{k}:{v}" for k, v in plan.character_actions.items())
    return AnimateResult(plan.id, AnimateStatus.APPLIED, summary)


def animate_chapter_shots(
    project_root: Path,
    chapter_num: int,
    *,
    blender: str | None = None,
    only_shots: set[str] | None = None,
    require_characters: set[str] | None = None,
    min_shot_number: int | None = None,
    quality: str = "daily",
) -> list[AnimateResult]:
    project_root = project_root.resolve()
    doc = load_shot_list_for_chapter(project_root, chapter_num)
    results: list[AnimateResult] = []

    for plan in doc.shots:
        if only_shots is not None and plan.id not in only_shots:
            continue
        if require_characters is not None:
            if not require_characters.intersection(plan.character_ids):
                continue
        if min_shot_number is not None:
            shot_num = int(plan.id.rsplit("_", maxsplit=1)[-1].removeprefix("shot"))
            if shot_num < min_shot_number:
                continue
        try:
            blend_path = _resolve_shot_blend(project_root, doc, plan.id)
        except FileNotFoundError as exc:
            results.append(AnimateResult(plan.id, AnimateStatus.FAILED, str(exc)))
            continue
        results.append(
            animate_shot(blend_path, plan, project_root, blender=blender, quality=quality)
        )

    return results
