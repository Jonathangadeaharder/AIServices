"""Hand-polish pass for shot animation from story mood and character roles."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from blenderservices.scaffold import load_shot_list_for_chapter, shots_dir
from blenderservices.shot_list import EpisodeShotPlan


class PolishStatus(StrEnum):
    POLISHED = "polished"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class PolishResult:
    shot_id: str
    status: PolishStatus
    message: str


def _mood_for_character(plan: EpisodeShotPlan, char_id: str) -> dict[str, float]:
    notes = plan.notes.lower()
    action = plan.character_actions.get(char_id, "idle")
    mood: dict[str, float] = {}

    if char_id == "gretel":
        if "weint" in notes or "schluchzte" in notes:
            mood.update(head_pitch=-14, spine_extra=10, shoulder_hunch=6, arm_damp=0.7)
        elif "nacht" in notes or "wurzeln" in notes:
            mood.update(head_pitch=-6, spine_extra=4, shoulder_hunch=3, arm_damp=0.85)
        elif "haus aus brot" in notes:
            mood.update(head_pitch=8, head_roll=2, arm_damp=0.9)
        elif action == "walk":
            mood.update(head_pitch=-3, head_roll=1.5, arm_damp=0.95)
        elif action == "tired_walk":
            mood.update(head_pitch=-8, spine_extra=6, shoulder_hunch=4, arm_damp=0.75)
        else:
            mood.update(head_pitch=-2, arm_damp=0.9)

    if char_id == "hansel":
        if "unter der decke" in notes or "schlief nicht" in notes or "kind in hansel" in notes:
            mood.update(head_pitch=4, spine_extra=-2, arm_damp=1.05)
        elif "hand auf die schwelle" in notes or action == "walk":
            mood.update(head_pitch=2, foot_lift=1.2, arm_damp=1.1)
        elif "haus aus brot" in notes:
            mood.update(head_pitch=10, spine_extra=-3, arm_damp=0.85)
        elif action == "tired_walk":
            mood.update(head_pitch=-2, spine_extra=3, arm_damp=0.9)
        elif "schluchzte" in notes:
            mood.update(head_pitch=1, shoulder_hunch=1, arm_damp=1.05)
        else:
            mood.update(head_pitch=1, foot_lift=1.05)

    if char_id == "crone":
        mood.update(head_pitch=6, spine_extra=14, shoulder_hunch=8, arm_damp=0.8, foot_lift=0.6)

    if char_id in {"father", "stepmother"}:
        if "weint" in notes or "loswerden" in notes or "nickt" in notes:
            mood.update(
                head_pitch=-8 if char_id == "father" else 4,
                shoulder_hunch=5,
                arm_damp=0.8,
            )
        elif "kessel" in notes or "beobachtet" in notes or "mehl" in notes:
            mood.update(
                head_pitch=3 if char_id == "stepmother" else -3,
                shoulder_hunch=4,
                arm_damp=0.85,
            )
        else:
            mood.update(
                head_pitch=-4 if char_id == "father" else 2, shoulder_hunch=3, arm_damp=0.85
            )

    return mood


def mood_for_shot(plan: EpisodeShotPlan) -> dict[str, dict[str, float]]:
    return {char_id: _mood_for_character(plan, char_id) for char_id in plan.character_ids}


def polish_shot(
    blend_path: Path,
    plan: EpisodeShotPlan,
    repo_root: Path,
    *,
    blender: str | None = None,
) -> PolishResult:
    if not plan.character_ids:
        return PolishResult(plan.id, PolishStatus.SKIPPED, "no characters")

    script = repo_root / "tools" / "polish_shot_actions.py"
    mood = mood_for_shot(plan)
    blender_bin = blender or "blender"
    cmd = [
        blender_bin,
        "--background",
        str(blend_path.resolve()),
        "--python",
        str(script),
        "--",
        "--characters",
        ",".join(plan.character_ids),
        "--mood",
        json.dumps(mood),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip()
        return PolishResult(plan.id, PolishStatus.FAILED, detail)
    return PolishResult(plan.id, PolishStatus.POLISHED, f"{len(plan.character_ids)} character(s)")


def polish_chapter_shots(
    project_root: Path,
    chapter_num: int,
    *,
    blender: str | None = None,
    only_shots: set[str] | None = None,
    min_shot_number: int | None = None,
) -> list[PolishResult]:
    project_root = project_root.resolve()
    doc = load_shot_list_for_chapter(project_root, chapter_num)
    results: list[PolishResult] = []

    for plan in doc.shots:
        if only_shots is not None and plan.id not in only_shots:
            continue
        if min_shot_number is not None:
            shot_num = int(plan.id.rsplit("_", maxsplit=1)[-1].removeprefix("shot"))
            if shot_num < min_shot_number:
                continue
        if not plan.character_ids:
            continue
        suffix = plan.id.rsplit("_", maxsplit=1)[-1]
        blend_path = shots_dir(project_root, doc) / f"{suffix}.blend"
        if not blend_path.is_file():
            results.append(PolishResult(plan.id, PolishStatus.FAILED, f"missing {blend_path.name}"))
            continue
        results.append(polish_shot(blend_path, plan, project_root, blender=blender))

    return results
