"""Asset library quality audit for Phase 2 signoff gates."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

from blenderservices.manifest import AssetStatus, AssetType, find_manifest_paths, load_manifest

REQUIRED_FACIAL_SHAPE_KEYS = (
    "brow_up",
    "brow_down",
    "eye_blink_l",
    "eye_blink_r",
    "eye_wide",
    "mouth_open",
    "smile",
    "frown",
    "mouth_o",
    "cheek_puff",
    "jaw_clench",
    "nose_scrunch",
)

REQUIRED_CAMERA_ANCHORS = ("cam_wide", "cam_medium", "cam_hero")
PREVIEW_DIR = "previews"


class FindingLevel(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class AuditFinding:
    asset_id: str
    manifest_path: Path
    level: FindingLevel
    message: str


@dataclass
class BlendInspection:
    asset_id: str
    asset_type: str
    camera_anchors: list[str] = field(default_factory=list)
    lighting_presets: list[str] = field(default_factory=list)
    facial_controls: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    missing_shape_keys: list[str] = field(default_factory=list)
    broken_library_links: list[str] = field(default_factory=list)
    symmetry_modifiers: list[str] = field(default_factory=list)
    missing_armature: bool = False


def discover_preview_files(asset_dir: Path) -> list[str]:
    preview_dir = asset_dir / PREVIEW_DIR
    if not preview_dir.is_dir():
        return []
    return sorted(f"previews/{path.name}" for path in preview_dir.glob("*.png"))


def audit_manifests(
    root: Path,
    inspections: dict[str, BlendInspection],
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for manifest_path in find_manifest_paths(root):
        if _is_shot_catalog_manifest(manifest_path, root):
            continue
        manifest = load_manifest(manifest_path)
        asset_dir = manifest_path.parent
        inspection = inspections.get(manifest.id)

        previews = discover_preview_files(asset_dir)
        if not previews:
            findings.append(
                AuditFinding(manifest.id, manifest_path, FindingLevel.WARN, "no preview PNGs")
            )
        elif not manifest.preview_files:
            findings.append(
                AuditFinding(
                    manifest.id,
                    manifest_path,
                    FindingLevel.WARN,
                    "previews exist on disk but manifest.preview_files is empty",
                )
            )

        blend_path = asset_dir / manifest.blend_file
        if not blend_path.is_file():
            findings.append(
                AuditFinding(
                    manifest.id,
                    manifest_path,
                    FindingLevel.FAIL,
                    f"missing blend_file: {manifest.blend_file}",
                )
            )
            continue

        if inspection is None:
            findings.append(
                AuditFinding(
                    manifest.id,
                    manifest_path,
                    FindingLevel.WARN,
                    "blend not inspected (Blender audit skipped or failed)",
                )
            )
            continue

        findings.extend(_character_findings(manifest_path, manifest, inspection))
        findings.extend(_set_findings(manifest_path, manifest, inspection))
        findings.extend(_shared_findings(manifest_path, manifest, inspection))

        if not any(f.level == FindingLevel.FAIL for f in findings if f.asset_id == manifest.id):
            if previews and manifest.status == AssetStatus.PLANNED:
                findings.append(
                    AuditFinding(
                        manifest.id,
                        manifest_path,
                        FindingLevel.PASS,
                        "eligible for review promotion",
                    )
                )
    return findings


def sync_manifest(
    manifest_path: Path,
    inspection: BlendInspection | None,
    *,
    promote_review: bool = True,
) -> bool:
    """Update manifest.yaml from disk previews and blend inspection. Returns True if written."""
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return False

    asset_dir = manifest_path.parent
    previews = discover_preview_files(asset_dir)
    changed = False

    if previews and data.get("preview_files") != previews:
        data["preview_files"] = previews
        changed = True

    if inspection is None:
        if changed:
            manifest_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return changed

    if inspection.asset_type == AssetType.SET.value:
        if inspection.camera_anchors and data.get("camera_anchors") != inspection.camera_anchors:
            data["camera_anchors"] = inspection.camera_anchors
            changed = True
        if (
            inspection.lighting_presets
            and data.get("lighting_presets") != inspection.lighting_presets
        ):
            data["lighting_presets"] = inspection.lighting_presets
            changed = True

    if inspection.asset_type == AssetType.CHARACTER.value:
        if data.get("facial_controls") != inspection.facial_controls:
            data["facial_controls"] = inspection.facial_controls
            changed = True
        if inspection.actions and data.get("actions") != inspection.actions:
            data["actions"] = inspection.actions
            changed = True
        if data.get("status") == AssetStatus.REVIEW.value and not data.get("actions"):
            data["status"] = AssetStatus.PLANNED.value
            changed = True
        for key, filename in (
            ("turntable_preview", "previews/turntable_front.png"),
            ("rig_test_preview", "previews/rig_test_pose.png"),
        ):
            if (asset_dir / filename).is_file() and data.get(key) != filename:
                data[key] = filename
                changed = True

    if inspection.asset_type == AssetType.ACTION_LIBRARY.value and inspection.actions:
        if data.get("actions") != inspection.actions:
            data["actions"] = inspection.actions
            changed = True

    if promote_review and previews and data.get("status") == AssetStatus.PLANNED.value:
        if inspection.asset_type == AssetType.CHARACTER.value and not data.get("actions"):
            pass
        elif _eligible_for_review(inspection, previews):
            data["status"] = AssetStatus.REVIEW.value
            changed = True

    if changed:
        manifest_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return changed


def run_blender_audit(root: Path, *, blender: str | None = None) -> dict[str, BlendInspection]:
    script = Path(__file__).resolve().parents[3] / "tools" / "audit_assets.py"
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
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = (result.stdout + result.stderr).strip()
        raise RuntimeError(f"Blender asset audit failed: {msg}")

    payload = _parse_audit_json(result.stdout)
    return {
        item["asset_id"]: BlendInspection(
            asset_id=item["asset_id"],
            asset_type=item["asset_type"],
            camera_anchors=item.get("camera_anchors", []),
            lighting_presets=item.get("lighting_presets", []),
            facial_controls=item.get("facial_controls", []),
            actions=item.get("actions", []),
            missing_shape_keys=item.get("missing_shape_keys", []),
            broken_library_links=item.get("broken_library_links", []),
            symmetry_modifiers=item.get("symmetry_modifiers", []),
            missing_armature=item.get("missing_armature", False),
        )
        for item in payload
    }


def audit_asset_tree(
    root: Path,
    *,
    blender: str | None = None,
    write: bool = False,
) -> tuple[list[AuditFinding], list[Path]]:
    inspections = run_blender_audit(root, blender=blender)
    written: list[Path] = []
    if write:
        for manifest_path in find_manifest_paths(root):
            if _is_shot_catalog_manifest(manifest_path, root):
                continue
            data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            asset_id = data.get("id")
            if not asset_id:
                continue
            if sync_manifest(manifest_path, inspections.get(asset_id)):
                written.append(manifest_path)
    findings = audit_manifests(root, inspections)
    return findings, written


def _character_findings(path: Path, manifest, inspection: BlendInspection) -> list[AuditFinding]:
    if manifest.asset_type != AssetType.CHARACTER:
        return []
    findings: list[AuditFinding] = []
    if inspection.missing_armature:
        findings.append(AuditFinding(manifest.id, path, FindingLevel.FAIL, "no armature found"))
    if inspection.missing_shape_keys:
        joined = ", ".join(inspection.missing_shape_keys)
        findings.append(
            AuditFinding(manifest.id, path, FindingLevel.FAIL, f"missing shape keys: {joined}")
        )
    if inspection.symmetry_modifiers:
        joined = ", ".join(inspection.symmetry_modifiers)
        findings.append(
            AuditFinding(
                manifest.id, path, FindingLevel.FAIL, f"symmetry modifiers active: {joined}"
            )
        )
    if manifest.status in {AssetStatus.REVIEW, AssetStatus.APPROVED} and not manifest.actions:
        findings.append(
            AuditFinding(
                manifest.id, path, FindingLevel.WARN, "review character missing manifest actions"
            )
        )
    return findings


def _set_findings(path: Path, manifest, inspection: BlendInspection) -> list[AuditFinding]:
    if manifest.asset_type != AssetType.SET:
        return []
    findings: list[AuditFinding] = []
    missing_cams = [c for c in REQUIRED_CAMERA_ANCHORS if c not in inspection.camera_anchors]
    if missing_cams:
        joined = ", ".join(missing_cams)
        findings.append(
            AuditFinding(manifest.id, path, FindingLevel.FAIL, f"missing camera anchors: {joined}")
        )
    if not inspection.lighting_presets:
        findings.append(
            AuditFinding(manifest.id, path, FindingLevel.WARN, "no lighting presets found in blend")
        )
    return findings


def _shared_findings(path: Path, manifest, inspection: BlendInspection) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for lib in inspection.broken_library_links:
        findings.append(
            AuditFinding(manifest.id, path, FindingLevel.FAIL, f"broken library: {lib}")
        )
    return findings


def _is_shot_catalog_manifest(manifest_path: Path, root: Path) -> bool:
    try:
        manifest_path.relative_to(root / "shots")
        return True
    except ValueError:
        return False


def _eligible_for_review(inspection: BlendInspection, previews: list[str]) -> bool:
    if not previews or inspection.broken_library_links:
        return False
    if inspection.asset_type == AssetType.CHARACTER.value:
        return not (
            inspection.missing_shape_keys
            or inspection.symmetry_modifiers
            or inspection.missing_armature
        )
    if inspection.asset_type == AssetType.SET.value:
        missing_cams = [c for c in REQUIRED_CAMERA_ANCHORS if c not in inspection.camera_anchors]
        return not missing_cams and bool(inspection.lighting_presets)
    if inspection.asset_type == AssetType.ACTION_LIBRARY.value:
        return bool(inspection.actions)
    return True


def _parse_audit_json(stdout: str) -> list[dict]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            return json.loads(line)
        if line.startswith("{") and line.endswith("}"):
            data = json.loads(line)
            return data.get("assets", [])
    raise RuntimeError("Blender asset audit produced no JSON payload")
