"""Manifest models and validation for the Blender-native 3D pipeline."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class AssetType(StrEnum):
    CHARACTER = "character"
    SET = "set"
    PROP = "prop"
    MATERIAL = "material"
    ACTION_LIBRARY = "action_library"
    SHOT = "shot"


class AssetStatus(StrEnum):
    PLANNED = "planned"
    PROXY = "proxy"
    WIP = "wip"
    REVIEW = "review"
    APPROVED = "approved"
    DEPRECATED = "deprecated"


ApprovedLicense = Literal["project-owned", "cc0", "public-domain"]


class BaseAssetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    id: str
    asset_type: AssetType
    status: AssetStatus
    blend_file: str
    preview_files: list[str] = Field(default_factory=list)
    license: ApprovedLicense
    notes: str = ""

    @model_validator(mode="after")
    def validate_review_ready_common_fields(self) -> BaseAssetManifest:
        if self.status in {AssetStatus.REVIEW, AssetStatus.APPROVED} and not self.preview_files:
            raise ValueError(f"{self.id}: review/approved assets require preview_files")
        return self


class CharacterManifest(BaseAssetManifest):
    asset_type: Literal[AssetType.CHARACTER]
    rig_collection: str
    actions: list[str] = Field(default_factory=list)
    facial_controls: list[str] = Field(default_factory=list)
    turntable_preview: str | None = None
    rig_test_preview: str | None = None

    @model_validator(mode="after")
    def validate_review_ready_character_fields(self) -> CharacterManifest:
        if self.status == AssetStatus.PROXY and not self.preview_files:
            raise ValueError(f"{self.id}: proxy character requires preview_files")
        if self.status in {AssetStatus.REVIEW, AssetStatus.APPROVED}:
            missing = []
            if not self.actions:
                missing.append("actions")
            if not self.turntable_preview:
                missing.append("turntable_preview")
            if not self.rig_test_preview:
                missing.append("rig_test_preview")
            if missing:
                msg = f"{self.id}: review/approved character missing {', '.join(missing)}"
                raise ValueError(msg)
        return self


class SetManifest(BaseAssetManifest):
    asset_type: Literal[AssetType.SET]
    set_collection: str
    camera_anchors: list[str] = Field(default_factory=list)
    lighting_presets: list[str] = Field(default_factory=list)
    dressing_density: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_review_ready_set_fields(self) -> SetManifest:
        if self.status in {AssetStatus.REVIEW, AssetStatus.APPROVED}:
            missing = []
            if not self.camera_anchors:
                missing.append("camera_anchors")
            if not self.lighting_presets:
                missing.append("lighting_presets")
            if missing:
                msg = f"{self.id}: review/approved set missing {', '.join(missing)}"
                raise ValueError(msg)
        return self


class PropManifest(BaseAssetManifest):
    asset_type: Literal[AssetType.PROP]
    prop_collection: str
    scale_reference: str | None = None


class MaterialManifest(BaseAssetManifest):
    asset_type: Literal[AssetType.MATERIAL]


class ActionLibraryManifest(BaseAssetManifest):
    asset_type: Literal[AssetType.ACTION_LIBRARY]
    actions: list[str] = Field(default_factory=list)
    compatible_character_ids: list[str] = Field(default_factory=list)
    frame_rate: int = 24

    @model_validator(mode="after")
    def validate_action_library_fields(self) -> ActionLibraryManifest:
        if self.frame_rate <= 0:
            raise ValueError(f"{self.id}: frame_rate must be positive")
        if self.status in {AssetStatus.REVIEW, AssetStatus.APPROVED} and not self.actions:
            raise ValueError(f"{self.id}: review/approved action library missing actions")
        return self


class ShotManifest(BaseAssetManifest):
    asset_type: Literal[AssetType.SHOT]
    set_id: str
    character_ids: list[str] = Field(default_factory=list)
    prop_ids: list[str] = Field(default_factory=list)
    action_library_ids: list[str] = Field(default_factory=list)
    character_actions: dict[str, str] = Field(default_factory=dict)
    camera: str | None = None
    lighting: str | None = None
    final_frame_sequence: str | None = None
    qc_status: Literal["pending", "pass", "waived", "fail"] = "pending"

    @model_validator(mode="after")
    def validate_shot_fields(self) -> ShotManifest:
        unknown_characters = sorted(set(self.character_actions) - set(self.character_ids))
        if unknown_characters:
            joined = ", ".join(unknown_characters)
            raise ValueError(f"{self.id}: character_actions reference unknown characters: {joined}")
        if self.status in {AssetStatus.REVIEW, AssetStatus.APPROVED}:
            missing = []
            if not self.camera:
                missing.append("camera")
            if not self.lighting:
                missing.append("lighting")
            if not self.final_frame_sequence:
                missing.append("final_frame_sequence")
            if self.qc_status not in {"pass", "waived"}:
                missing.append("qc_status pass/waived")
            if missing:
                msg = f"{self.id}: review/approved shot missing {', '.join(missing)}"
                raise ValueError(msg)
        return self


Manifest = (
    CharacterManifest
    | SetManifest
    | PropManifest
    | MaterialManifest
    | ActionLibraryManifest
    | ShotManifest
)


_MANIFEST_BY_TYPE: dict[str, type[Manifest]] = {
    AssetType.CHARACTER.value: CharacterManifest,
    AssetType.SET.value: SetManifest,
    AssetType.PROP.value: PropManifest,
    AssetType.MATERIAL.value: MaterialManifest,
    AssetType.ACTION_LIBRARY.value: ActionLibraryManifest,
    AssetType.SHOT.value: ShotManifest,
}


class ManifestIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    message: str


def load_manifest(path: Path) -> Manifest:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: manifest must be a mapping")
    asset_type = str(data.get("asset_type", ""))
    manifest_type = _MANIFEST_BY_TYPE.get(asset_type)
    if manifest_type is None:
        raise ValueError(f"{path}: unknown asset_type {asset_type!r}")
    return manifest_type.model_validate(data)


def find_manifest_paths(root: Path) -> list[Path]:
    return sorted(root.glob("**/manifest.yaml"))


def validate_asset_tree(root: Path) -> list[ManifestIssue]:
    issues: list[ManifestIssue] = []
    if not root.exists():
        return [ManifestIssue(path=root, message="asset root does not exist")]
    manifests_by_id: dict[str, tuple[Path, Manifest]] = {}
    duplicate_ids: dict[str, list[Path]] = {}
    for path in find_manifest_paths(root):
        try:
            manifest = load_manifest(path)
        except (OSError, ValueError, ValidationError) as exc:
            issues.append(ManifestIssue(path=path, message=str(exc)))
            continue
        if manifest.id in manifests_by_id:
            duplicate_ids.setdefault(manifest.id, [manifests_by_id[manifest.id][0]]).append(path)
        else:
            manifests_by_id[manifest.id] = (path, manifest)
        issues.extend(_validate_manifest_files(path, manifest))
        issues.extend(_validate_manifest_contract(path, manifest))
    for asset_id, paths in sorted(duplicate_ids.items()):
        rendered_paths = ", ".join(str(path) for path in paths)
        issues.append(
            ManifestIssue(
                path=root,
                message=f"duplicate asset id {asset_id}: {rendered_paths}",
            )
        )
    for path, manifest in manifests_by_id.values():
        if isinstance(manifest, ShotManifest):
            issues.extend(_validate_shot_references(path, manifest, manifests_by_id))
    return issues


def _validate_manifest_files(path: Path, manifest: Manifest) -> list[ManifestIssue]:
    issues: list[ManifestIssue] = []
    asset_dir = path.parent
    blend_path = asset_dir / manifest.blend_file
    if manifest.status != AssetStatus.PLANNED and not blend_path.is_file():
        issues.append(
            ManifestIssue(path=path, message=f"missing blend_file: {manifest.blend_file}")
        )

    for preview in manifest.preview_files:
        if not (asset_dir / preview).is_file():
            issues.append(ManifestIssue(path=path, message=f"missing preview file: {preview}"))

    for attr in ("turntable_preview", "rig_test_preview", "final_frame_sequence"):
        value = getattr(manifest, attr, None)
        if isinstance(value, str) and value and not (asset_dir / value).exists():
            issues.append(ManifestIssue(path=path, message=f"missing {attr}: {value}"))
    return issues


def _validate_manifest_contract(path: Path, manifest: Manifest) -> list[ManifestIssue]:
    issues: list[ManifestIssue] = []
    if isinstance(manifest, CharacterManifest) and manifest.rig_collection != f"CHAR_{manifest.id}":
        issues.append(
            ManifestIssue(
                path=path,
                message=f"rig_collection must be CHAR_{manifest.id}",
            )
        )
    if isinstance(manifest, SetManifest) and manifest.set_collection != f"SET_{manifest.id}":
        issues.append(
            ManifestIssue(
                path=path,
                message=f"set_collection must be SET_{manifest.id}",
            )
        )
    if isinstance(manifest, PropManifest) and manifest.prop_collection != f"PROP_{manifest.id}":
        issues.append(
            ManifestIssue(
                path=path,
                message=f"prop_collection must be PROP_{manifest.id}",
            )
        )
    return issues


def _validate_shot_references(
    path: Path,
    shot: ShotManifest,
    manifests_by_id: dict[str, tuple[Path, Manifest]],
) -> list[ManifestIssue]:
    issues: list[ManifestIssue] = []
    set_manifest = _reference_manifest(path, shot.set_id, AssetType.SET, manifests_by_id, issues)
    if isinstance(set_manifest, SetManifest):
        if (
            shot.camera
            and set_manifest.camera_anchors
            and shot.camera not in set_manifest.camera_anchors
        ):
            issues.append(
                ManifestIssue(
                    path=path,
                    message=f"unknown camera anchor {shot.camera!r} for set {shot.set_id}",
                )
            )
        if (
            shot.lighting
            and set_manifest.lighting_presets
            and shot.lighting not in set_manifest.lighting_presets
        ):
            issues.append(
                ManifestIssue(
                    path=path,
                    message=f"unknown lighting preset {shot.lighting!r} for set {shot.set_id}",
                )
            )

    for character_id in shot.character_ids:
        _reference_manifest(path, character_id, AssetType.CHARACTER, manifests_by_id, issues)
    for prop_id in shot.prop_ids:
        _reference_manifest(path, prop_id, AssetType.PROP, manifests_by_id, issues)
    action_names_by_library: dict[str, set[str]] = {}
    for library_id in shot.action_library_ids:
        library = _reference_manifest(
            path,
            library_id,
            AssetType.ACTION_LIBRARY,
            manifests_by_id,
            issues,
        )
        if isinstance(library, ActionLibraryManifest):
            action_names_by_library[library_id] = set(library.actions)
            incompatible = sorted(set(shot.character_ids) - set(library.compatible_character_ids))
            if library.compatible_character_ids and incompatible:
                joined = ", ".join(incompatible)
                issues.append(
                    ManifestIssue(
                        path=path,
                        message=f"action library {library_id} is not compatible with: {joined}",
                    )
                )
    available_actions = (
        set().union(*action_names_by_library.values()) if action_names_by_library else set()
    )
    for character_id, action_name in sorted(shot.character_actions.items()):
        if available_actions and action_name not in available_actions:
            issues.append(
                ManifestIssue(
                    path=path,
                    message=f"unknown action {action_name!r} assigned to {character_id}",
                )
            )
    return issues


def _reference_manifest(
    path: Path,
    asset_id: str,
    expected_type: AssetType,
    manifests_by_id: dict[str, tuple[Path, Manifest]],
    issues: list[ManifestIssue],
) -> Manifest | None:
    entry = manifests_by_id.get(asset_id)
    if entry is None:
        issues.append(
            ManifestIssue(
                path=path,
                message=f"missing referenced {expected_type.value}: {asset_id}",
            )
        )
        return None
    _ref_path, manifest = entry
    if manifest.asset_type != expected_type:
        issues.append(
            ManifestIssue(
                path=path,
                message=(
                    f"referenced asset {asset_id} must be {expected_type.value}, "
                    f"got {manifest.asset_type.value}"
                ),
            )
        )
    return manifest


def manifest_summary(root: Path) -> dict[str, Any]:
    manifests: list[Manifest] = []
    for path in find_manifest_paths(root):
        manifests.append(load_manifest(path))
    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for manifest in manifests:
        by_type[manifest.asset_type.value] = by_type.get(manifest.asset_type.value, 0) + 1
        by_status[manifest.status.value] = by_status.get(manifest.status.value, 0) + 1
    return {
        "total": len(manifests),
        "by_type": dict(sorted(by_type.items())),
        "by_status": dict(sorted(by_status.items())),
    }
