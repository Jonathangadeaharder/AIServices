"""Hero-shot production pass: setup, animation, full render, QC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from blenderservices.render_qc import (
    QCStatus,
    assemble_chapter_from_renders,
    default_chapter_narration_path,
    render_shot_mp4,
    render_shot_qc,
)
from blenderservices.scaffold import load_shot_list_for_chapter, shots_dir
from blenderservices.shot_animation import AnimateStatus
from blenderservices.shot_polish import PolishStatus, polish_shot
from blenderservices.shot_setup import (
    ApplyStatus,
    apply_shot_from_plan,
    apply_story_actions,
    stage_shot_layout,
)


@dataclass(frozen=True)
class HeroShotResult:
    shot_id: str
    setup: ApplyStatus
    animation: AnimateStatus
    polish: PolishStatus
    render_path: Path | None
    qc_status: QCStatus | None
    message: str


def run_hero_shot(
    project_root: Path,
    chapter_num: int,
    shot_id: str,
    *,
    blender: str | None = None,
    reassemble_chapter: bool = True,
) -> HeroShotResult:
    project_root = project_root.resolve()
    doc = load_shot_list_for_chapter(project_root, chapter_num)
    plan = next((shot for shot in doc.shots if shot.id == shot_id), None)
    if plan is None:
        msg = f"unknown shot id: {shot_id}"
        raise ValueError(msg)

    suffix = shot_id.rsplit("_", maxsplit=1)[-1]
    blend_path = shots_dir(project_root, doc) / f"{suffix}.blend"
    if not blend_path.is_file():
        msg = f"missing blend for {shot_id}: {blend_path}"
        raise FileNotFoundError(msg)

    setup = apply_shot_from_plan(
        blend_path,
        plan,
        project_root,
        blender=blender,
        character_source="story",
        action_source="story",
    )
    if setup.status == ApplyStatus.FAILED:
        return HeroShotResult(
            shot_id,
            setup.status,
            AnimateStatus.SKIPPED,
            PolishStatus.SKIPPED,
            None,
            None,
            setup.message,
        )

    animation = apply_story_actions(blend_path, plan, project_root, blender=blender, quality="hero")
    if animation.status == AnimateStatus.FAILED:
        return HeroShotResult(
            shot_id,
            setup.status,
            animation.status,
            PolishStatus.SKIPPED,
            None,
            None,
            animation.message,
        )

    # Hero animate_shot re-keys every bone with the Mixamo idle action, which
    # wipes the per-shot acting / arm pose overlay applied earlier inside
    # apply_shot_from_plan. Re-stage now to re-purge those fcurves so the pose
    # persists into the hero render.
    restage = stage_shot_layout(blend_path, plan, project_root, blender=blender)
    if restage.status == ApplyStatus.FAILED:
        return HeroShotResult(
            shot_id,
            setup.status,
            animation.status,
            PolishStatus.SKIPPED,
            None,
            None,
            restage.message,
        )

    hand_polish = polish_shot(blend_path, plan, project_root, blender=blender)
    if hand_polish.status == PolishStatus.FAILED:
        return HeroShotResult(
            shot_id,
            setup.status,
            animation.status,
            hand_polish.status,
            None,
            None,
            hand_polish.message,
        )

    render = render_shot_mp4(project_root, doc, shot_id, frame_end=None, blender=blender)
    qc = render_shot_qc(project_root, doc, shot_id, blender=blender)

    if reassemble_chapter:
        narration = default_chapter_narration_path(project_root, chapter_num)
        output = project_root / "build" / "renders" / doc.episode / f"chapter_{chapter_num:02d}.mp4"
        assemble_chapter_from_renders(
            project_root,
            chapter_num,
            output,
            narration_path=narration if narration.is_file() else None,
        )

    return HeroShotResult(
        shot_id,
        setup.status,
        animation.status,
        hand_polish.status,
        render.output,
        qc.status,
        render.message if render.output else render.message,
    )


def run_hero_chapter(
    project_root: Path,
    chapter_num: int,
    *,
    blender: str | None = None,
    skip_shots: set[str] | None = None,
    only_characters: bool = False,
) -> list[HeroShotResult]:
    """Full-length hero pass for every chapter shot, then assemble once."""
    project_root = project_root.resolve()
    doc = load_shot_list_for_chapter(project_root, chapter_num)
    skip = skip_shots or set()
    results: list[HeroShotResult] = []

    for plan in doc.shots:
        if plan.id in skip:
            continue
        if only_characters and not plan.character_actions:
            continue

        if plan.character_actions:
            results.append(
                run_hero_shot(
                    project_root,
                    chapter_num,
                    plan.id,
                    blender=blender,
                    reassemble_chapter=False,
                )
            )
            continue

        suffix = plan.id.rsplit("_", maxsplit=1)[-1]
        blend_path = shots_dir(project_root, doc) / f"{suffix}.blend"
        if not blend_path.is_file():
            results.append(
                HeroShotResult(
                    plan.id,
                    ApplyStatus.SKIPPED,
                    AnimateStatus.SKIPPED,
                    PolishStatus.SKIPPED,
                    None,
                    None,
                    f"missing {blend_path.name}",
                )
            )
            continue

        apply_shot_from_plan(
            blend_path,
            plan,
            project_root,
            blender=blender,
            character_source="story",
            action_source="story",
        )
        render = render_shot_mp4(project_root, doc, plan.id, frame_end=None, blender=blender)
        qc = render_shot_qc(project_root, doc, plan.id, blender=blender)
        results.append(
            HeroShotResult(
                plan.id,
                ApplyStatus.APPLIED,
                AnimateStatus.SKIPPED,
                PolishStatus.SKIPPED,
                render.output,
                qc.status,
                render.message if render.output else render.message,
            )
        )

    narration = default_chapter_narration_path(project_root, chapter_num)
    output = project_root / "build" / "renders" / doc.episode / f"chapter_{chapter_num:02d}.mp4"
    assemble_chapter_from_renders(
        project_root,
        chapter_num,
        output,
        narration_path=narration if narration.is_file() else None,
    )
    return results
