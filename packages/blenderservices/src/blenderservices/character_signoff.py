"""Phase 5 character signoff gates for reusable production rigs."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import yaml

from blenderservices.manifest import AssetStatus

CHILD_CHARACTER_IDS = ("gretel", "hansel")
ADULT_CHARACTER_IDS = ("crone", "father", "stepmother")
ALL_CHARACTER_IDS = CHILD_CHARACTER_IDS + ADULT_CHARACTER_IDS

REFERENCE_PLATE_SOURCES = {
    "reference/front.png": "previews/turntable_front.png",
    "reference/side.png": "previews/turntable_side.png",
    "reference/back.png": "previews/turntable_back.png",
    "reference/three_quarter.png": "previews/turntable_3q.png",
}

REQUIRED_CHILD_ACTIONS = (
    "idle",
    "walk",
    "run",
    "tired_walk",
    "fear",
    "cry",
    "reach",
    "kneel",
    "listen",
)

REQUIRED_ADULT_BASE_ACTIONS = (
    "idle",
    "walk",
    "tired_walk",
    "reach",
    "listen",
    "fear",
    "kneel",
)

REQUIRED_RETARGET_BONES = (
    "Root",
    "Hips",
    "Spine01",
    "Spine02",
    "Head",
    "UpperArm.L",
    "ForeArm.L",
    "UpperArm.R",
    "ForeArm.R",
    "UpperLeg.L",
    "LowerLeg.L",
    "Foot.L",
    "UpperLeg.R",
    "LowerLeg.R",
    "Foot.R",
)

SIGNOFF_NOTE = (
    "Phase 5 signoff: reference plates, retarget bones, facial shape keys, "
    "and required actions validated."
)


class SignoffProfile(NamedTuple):
    actions: tuple[str, ...]
    height_range: tuple[float, float]


CHARACTER_SIGNOFF_PROFILES: dict[str, SignoffProfile] = {
    "gretel": SignoffProfile(REQUIRED_CHILD_ACTIONS, (1.15, 1.85)),
    "hansel": SignoffProfile(REQUIRED_CHILD_ACTIONS, (1.15, 1.85)),
    "crone": SignoffProfile(
        REQUIRED_ADULT_BASE_ACTIONS + ("push",),
        (1.65, 2.50),
    ),
    "father": SignoffProfile(
        REQUIRED_ADULT_BASE_ACTIONS + ("carry", "grief"),
        (1.70, 2.10),
    ),
    "stepmother": SignoffProfile(
        REQUIRED_ADULT_BASE_ACTIONS + ("fury", "sneer"),
        (1.60, 2.00),
    ),
}


class SignoffLevel(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class SignoffFinding:
    character_id: str
    level: SignoffLevel
    message: str


@dataclass
class CharacterInspection:
    character_id: str
    armature_name: str = ""
    rig_collection: str = ""
    bone_names: list[str] = field(default_factory=list)
    facial_controls: list[str] = field(default_factory=list)
    missing_shape_keys: list[str] = field(default_factory=list)
    missing_retarget_bones: list[str] = field(default_factory=list)
    symmetry_modifiers: list[str] = field(default_factory=list)
    mesh_height_m: float = 0.0
    missing_armature: bool = False
    missing_rig_collection: bool = False


def required_actions_for(character_id: str) -> tuple[str, ...]:
    profile = CHARACTER_SIGNOFF_PROFILES.get(character_id)
    if profile is None:
        raise ValueError(f"unknown character id: {character_id}")
    return profile.actions


def character_asset_dir(root: Path, character_id: str) -> Path:
    return root / "characters" / character_id


def sync_reference_plates(asset_dir: Path) -> list[Path]:
    written: list[Path] = []
    for dest_rel, source_rel in REFERENCE_PLATE_SOURCES.items():
        source = asset_dir / source_rel
        dest = asset_dir / dest_rel
        if not source.is_file():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.is_file() or source.stat().st_mtime_ns > dest.stat().st_mtime_ns:
            shutil.copy2(source, dest)
            written.append(dest)
    return written


def evaluate_signoff(  # noqa: C901
    character_id: str,
    inspection: CharacterInspection,
    *,
    required_actions: tuple[str, ...],
    manifest_actions: list[str],
    reference_ready: bool,
    preview_ready: bool,
    height_range: tuple[float, float] | None = None,
) -> list[SignoffFinding]:
    findings: list[SignoffFinding] = []

    if inspection.missing_armature:
        findings.append(SignoffFinding(character_id, SignoffLevel.FAIL, "no armature found"))
    if inspection.missing_rig_collection:
        findings.append(SignoffFinding(character_id, SignoffLevel.FAIL, "rig collection missing"))
    if not preview_ready:
        findings.append(SignoffFinding(character_id, SignoffLevel.FAIL, "preview PNGs missing"))
    if not reference_ready:
        findings.append(SignoffFinding(character_id, SignoffLevel.FAIL, "reference plates missing"))

    missing_actions = [action for action in required_actions if action not in manifest_actions]
    if missing_actions:
        joined = ", ".join(missing_actions)
        findings.append(
            SignoffFinding(character_id, SignoffLevel.FAIL, f"missing actions: {joined}")
        )

    if inspection.missing_retarget_bones:
        joined = ", ".join(inspection.missing_retarget_bones)
        findings.append(
            SignoffFinding(character_id, SignoffLevel.FAIL, f"missing retarget bones: {joined}")
        )
    if inspection.missing_shape_keys:
        joined = ", ".join(inspection.missing_shape_keys)
        findings.append(
            SignoffFinding(character_id, SignoffLevel.FAIL, f"missing shape keys: {joined}")
        )
    if inspection.symmetry_modifiers:
        joined = ", ".join(inspection.symmetry_modifiers)
        findings.append(
            SignoffFinding(character_id, SignoffLevel.FAIL, f"symmetry modifiers active: {joined}")
        )

    if height_range and inspection.mesh_height_m:
        low, high = height_range
        if inspection.mesh_height_m < low or inspection.mesh_height_m > high:
            findings.append(
                SignoffFinding(
                    character_id,
                    SignoffLevel.WARN,
                    f"mesh height {inspection.mesh_height_m:.2f}m outside expected range "
                    f"{low:.2f}-{high:.2f}m",
                )
            )

    if not any(f.level == SignoffLevel.FAIL for f in findings):
        findings.append(SignoffFinding(character_id, SignoffLevel.PASS, "ready for approval"))
    return findings


def apply_signoff(  # noqa: C901
    manifest_path: Path,
    inspection: CharacterInspection,
    *,
    approve: bool = False,
) -> tuple[bool, list[Path]]:
    asset_dir = manifest_path.parent
    reference_written = sync_reference_plates(asset_dir)

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return False, reference_written
    character_id = str(data.get("id") or inspection.character_id)
    profile = CHARACTER_SIGNOFF_PROFILES[character_id]
    required_actions = profile.actions
    changed = False

    previews = (
        sorted(f"previews/{path.name}" for path in (asset_dir / "previews").glob("*.png"))
        if (asset_dir / "previews").is_dir()
        else []
    )
    if previews and data.get("preview_files") != previews:
        data["preview_files"] = previews
        changed = True

    if list(data.get("actions") or []) != list(required_actions):
        data["actions"] = list(required_actions)
        changed = True

    if inspection.facial_controls and data.get("facial_controls") != inspection.facial_controls:
        data["facial_controls"] = inspection.facial_controls
        changed = True

    for key, filename in (
        ("turntable_preview", "previews/turntable_front.png"),
        ("rig_test_preview", "previews/rig_test_pose.png"),
    ):
        if (asset_dir / filename).is_file() and data.get(key) != filename:
            data[key] = filename
            changed = True

    reference_ready = all((asset_dir / rel).is_file() for rel in REFERENCE_PLATE_SOURCES)
    findings = evaluate_signoff(
        character_id,
        inspection,
        required_actions=required_actions,
        manifest_actions=data.get("actions", []),
        reference_ready=reference_ready,
        preview_ready=bool(previews),
        height_range=profile.height_range,
    )
    can_approve = approve and not any(f.level == SignoffLevel.FAIL for f in findings)

    if can_approve and data.get("status") != AssetStatus.APPROVED.value:
        data["status"] = AssetStatus.APPROVED.value
        existing = str(data.get("notes") or "").strip()
        if SIGNOFF_NOTE not in existing:
            data["notes"] = f"{existing} {SIGNOFF_NOTE}".strip() if existing else SIGNOFF_NOTE
        changed = True
    elif (
        not can_approve
        and data.get("status") == AssetStatus.PLANNED.value
        and reference_ready
        and previews
    ):
        data["status"] = AssetStatus.REVIEW.value
        changed = True

    if changed:
        manifest_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return changed, reference_written


def run_blender_signoff(
    root: Path, character_ids: list[str], *, blender: str | None = None
) -> dict[str, CharacterInspection]:
    script = Path(__file__).resolve().parents[3] / "tools" / "signoff_character.py"
    if not script.is_file():
        return {}

    blender_bin = blender or "blender"
    cmd = [
        blender_bin,
        "--background",
        "--factory-startup",
        "--python",
        str(script),
        "--",
        "--root",
        str(root.resolve()),
        "--characters",
        ",".join(character_ids),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = (result.stdout + result.stderr).strip()
        raise RuntimeError(f"Blender character signoff failed: {msg}")

    payload = _parse_signoff_json(result.stdout)
    return {
        item["character_id"]: CharacterInspection(
            character_id=item["character_id"],
            armature_name=item.get("armature_name", ""),
            rig_collection=item.get("rig_collection", ""),
            bone_names=item.get("bone_names", []),
            facial_controls=item.get("facial_controls", []),
            missing_shape_keys=item.get("missing_shape_keys", []),
            missing_retarget_bones=item.get("missing_retarget_bones", []),
            symmetry_modifiers=item.get("symmetry_modifiers", []),
            mesh_height_m=item.get("mesh_height_m", 0.0),
            missing_armature=item.get("missing_armature", False),
            missing_rig_collection=item.get("missing_rig_collection", False),
        )
        for item in payload
    }


def signoff_characters(
    root: Path,
    character_ids: list[str],
    *,
    blender: str | None = None,
    write: bool = False,
    approve: bool = False,
) -> tuple[list[SignoffFinding], list[Path], list[Path]]:
    unknown = [
        character_id
        for character_id in character_ids
        if character_id not in CHARACTER_SIGNOFF_PROFILES
    ]
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"unsupported character ids: {joined}")

    inspections = run_blender_signoff(root, character_ids, blender=blender)
    findings: list[SignoffFinding] = []
    written_manifests: list[Path] = []
    written_references: list[Path] = []

    for character_id in character_ids:
        asset_dir = character_asset_dir(root, character_id)
        manifest_path = asset_dir / "manifest.yaml"
        profile = CHARACTER_SIGNOFF_PROFILES[character_id]
        inspection = inspections.get(
            character_id,
            CharacterInspection(character_id=character_id, missing_armature=True),
        )

        if write and manifest_path.is_file():
            changed, refs = apply_signoff(manifest_path, inspection, approve=approve)
            if changed:
                written_manifests.append(manifest_path)
            written_references.extend(refs)
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            actions = manifest.get("actions", [])
        elif manifest_path.is_file():
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            actions = manifest.get("actions", list(profile.actions))
        else:
            actions = list(profile.actions)

        reference_ready = all(
            (asset_dir / rel).is_file() for rel in REFERENCE_PLATE_SOURCES
        ) or all((asset_dir / src).is_file() for src in REFERENCE_PLATE_SOURCES.values())
        preview_ready = (asset_dir / "previews").is_dir() and any(
            (asset_dir / "previews").glob("*.png")
        )
        findings.extend(
            evaluate_signoff(
                character_id,
                inspection,
                required_actions=profile.actions,
                manifest_actions=actions,
                reference_ready=reference_ready,
                preview_ready=preview_ready,
                height_range=profile.height_range,
            )
        )

    return findings, written_manifests, written_references


def _parse_signoff_json(stdout: str) -> list[dict]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            return json.loads(line)
    raise RuntimeError("Blender character signoff produced no JSON payload")
