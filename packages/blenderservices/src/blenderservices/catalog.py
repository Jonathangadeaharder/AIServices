"""Production catalog for reusable Gretel 3D assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

CatalogType = Literal["character", "set", "prop", "material", "action_library"]


@dataclass(frozen=True)
class CatalogAsset:
    id: str
    asset_type: CatalogType
    blend_file: str
    license: str = "project-owned"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def manifest_dir(self) -> Path:
        plural = {
            "character": "characters",
            "set": "sets",
            "prop": "props",
            "material": "materials",
            "action_library": "actions",
        }[self.asset_type]
        return Path(plural) / self.id

    def to_manifest(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": "1.0",
            "id": self.id,
            "asset_type": self.asset_type,
            "status": "planned",
            "blend_file": self.blend_file,
            "preview_files": [],
            "license": self.license,
            "notes": self.notes,
        }
        data.update(self.extra)
        return data


COMMON_CHILD_ACTIONS = [
    "idle",
    "walk",
    "tired_walk",
    "run",
    "stumble",
    "fear",
    "listen",
    "cry",
    "reach",
    "kneel",
]


PRODUCTION_CATALOG: tuple[CatalogAsset, ...] = (
    CatalogAsset(
        id="gretel",
        asset_type="character",
        blend_file="gretel.blend",
        notes="Primary prototype child rig for the 3D pipeline.",
        extra={
            "rig_collection": "CHAR_gretel",
            "actions": [],
            "facial_controls": [],
            "turntable_preview": None,
            "rig_test_preview": None,
        },
    ),
    CatalogAsset(
        id="hansel",
        asset_type="character",
        blend_file="hansel.blend",
        notes="Second child rig; must remain scale-compatible with Gretel.",
        extra={
            "rig_collection": "CHAR_hansel",
            "actions": [],
            "facial_controls": [],
            "turntable_preview": None,
            "rig_test_preview": None,
        },
    ),
    CatalogAsset(
        id="father",
        asset_type="character",
        blend_file="father.blend",
        notes="Gaunt woodcutter father rig for hut and forest staging.",
        extra={
            "rig_collection": "CHAR_father",
            "actions": [],
            "facial_controls": [],
            "turntable_preview": None,
            "rig_test_preview": None,
        },
    ),
    CatalogAsset(
        id="stepmother",
        asset_type="character",
        blend_file="stepmother.blend",
        notes="Hard, famine-worn stepmother rig for dialogue and departure scenes.",
        extra={
            "rig_collection": "CHAR_stepmother",
            "actions": [],
            "facial_controls": [],
            "turntable_preview": None,
            "rig_test_preview": None,
        },
    ),
    CatalogAsset(
        id="crone",
        asset_type="character",
        blend_file="crone.blend",
        notes="Crone rig for close-up menace, cellar, oven, and struggle beats.",
        extra={
            "rig_collection": "CHAR_crone",
            "actions": [],
            "facial_controls": [],
            "turntable_preview": None,
            "rig_test_preview": None,
        },
    ),
    CatalogAsset(
        id="hut_interior",
        asset_type="set",
        blend_file="hut_interior.blend",
        notes="Family hut interior with hearth, table, beds, rain leaks, and famine props.",
        extra={
            "set_collection": "SET_hut_interior",
            "camera_anchors": [],
            "lighting_presets": [],
            "dressing_density": ["low", "medium", "hero"],
        },
    ),
    CatalogAsset(
        id="hut_exterior",
        asset_type="set",
        blend_file="hut_exterior.blend",
        notes="Crooked hut exterior, muddy yard, rain, and forest edge approach.",
        extra={
            "set_collection": "SET_hut_exterior",
            "camera_anchors": [],
            "lighting_presets": [],
            "dressing_density": ["low", "medium", "hero"],
        },
    ),
    CatalogAsset(
        id="forest_edge",
        asset_type="set",
        blend_file="forest_edge.blend",
        notes="Transition set where hut world gives way to dark pines.",
        extra={
            "set_collection": "SET_forest_edge",
            "camera_anchors": [],
            "lighting_presets": [],
            "dressing_density": ["low", "medium", "hero"],
        },
    ),
    CatalogAsset(
        id="forest_path",
        asset_type="set",
        blend_file="forest_path.blend",
        notes="Primary forest path prototype for camera movement and modular dressing.",
        extra={
            "set_collection": "SET_forest_path",
            "camera_anchors": [],
            "lighting_presets": [],
            "dressing_density": ["low", "medium", "hero"],
        },
    ),
    CatalogAsset(
        id="deep_forest",
        asset_type="set",
        blend_file="deep_forest.blend",
        notes="Dense forest variant for lost, fearful, and predatory beats.",
        extra={
            "set_collection": "SET_deep_forest",
            "camera_anchors": [],
            "lighting_presets": [],
            "dressing_density": ["low", "medium", "hero"],
        },
    ),
    CatalogAsset(
        id="root_shelter",
        asset_type="set",
        blend_file="root_shelter.blend",
        notes="Hollow root shelter for overnight forest beats.",
        extra={
            "set_collection": "SET_root_shelter",
            "camera_anchors": [],
            "lighting_presets": [],
            "dressing_density": ["low", "medium", "hero"],
        },
    ),
    CatalogAsset(
        id="gingerbread_house_exterior",
        asset_type="set",
        blend_file="gingerbread_house_exterior.blend",
        notes="Practical edible house exterior; grim, tempting, not candy-bright.",
        extra={
            "set_collection": "SET_gingerbread_house_exterior",
            "camera_anchors": [],
            "lighting_presets": [],
            "dressing_density": ["low", "medium", "hero"],
        },
    ),
    CatalogAsset(
        id="gingerbread_house_interior",
        asset_type="set",
        blend_file="gingerbread_house_interior.blend",
        notes="Crone house interior with abundance, danger, and oven-room transitions.",
        extra={
            "set_collection": "SET_gingerbread_house_interior",
            "camera_anchors": [],
            "lighting_presets": [],
            "dressing_density": ["low", "medium", "hero"],
        },
    ),
    CatalogAsset(
        id="oven_room",
        asset_type="set",
        blend_file="oven_room.blend",
        notes="Oven staging set for firelight, menace, and crone struggle beats.",
        extra={
            "set_collection": "SET_oven_room",
            "camera_anchors": [],
            "lighting_presets": [],
            "dressing_density": ["low", "medium", "hero"],
        },
    ),
    CatalogAsset(
        id="cellar",
        asset_type="set",
        blend_file="cellar.blend",
        notes="Underground holding and fear set with shackles, dust, and hard light.",
        extra={
            "set_collection": "SET_cellar",
            "camera_anchors": [],
            "lighting_presets": [],
            "dressing_density": ["low", "medium", "hero"],
        },
    ),
    CatalogAsset(
        id="river_crossing",
        asset_type="set",
        blend_file="river_crossing.blend",
        notes="River set for return journey, duck crossing, and release beats.",
        extra={
            "set_collection": "SET_river_crossing",
            "camera_anchors": [],
            "lighting_presets": [],
            "dressing_density": ["low", "medium", "hero"],
        },
    ),
    CatalogAsset(
        id="castle_gate",
        asset_type="set",
        blend_file="castle_gate.blend",
        notes="Castle approach and gate set for ending contrast and scale.",
        extra={
            "set_collection": "SET_castle_gate",
            "camera_anchors": [],
            "lighting_presets": [],
            "dressing_density": ["low", "medium", "hero"],
        },
    ),
    CatalogAsset(
        id="castle_interior",
        asset_type="set",
        blend_file="castle_interior.blend",
        notes="Final interior set for safety, wealth, and story closure.",
        extra={
            "set_collection": "SET_castle_interior",
            "camera_anchors": [],
            "lighting_presets": [],
            "dressing_density": ["low", "medium", "hero"],
        },
    ),
    CatalogAsset(
        id="engraved_woodcut",
        asset_type="material",
        blend_file="engraved_woodcut.blend",
        notes="Shared procedural shader and compositor target for Engraved Stop-Motion 3D.",
    ),
    CatalogAsset(
        id="shared_locomotion",
        asset_type="action_library",
        blend_file="shared_locomotion.blend",
        notes="Reusable locomotion actions for primary child rigs.",
        extra={
            "actions": COMMON_CHILD_ACTIONS,
            "compatible_character_ids": ["gretel", "hansel"],
            "frame_rate": 24,
        },
    ),
    CatalogAsset(
        id="stones",
        asset_type="prop",
        blend_file="stones.blend",
        notes="Reusable stones for trail-marking and hand interaction.",
        extra={"prop_collection": "PROP_stones", "scale_reference": "child-hand-sized"},
    ),
    CatalogAsset(
        id="bread_crumbs",
        asset_type="prop",
        blend_file="bread_crumbs.blend",
        notes="Reusable crumb clusters with scale variants for forest path shots.",
        extra={"prop_collection": "PROP_bread_crumbs", "scale_reference": "finger-sized"},
    ),
    CatalogAsset(
        id="bowl",
        asset_type="prop",
        blend_file="bowl.blend",
        notes="Wooden bowl for hut poverty and food scarcity staging.",
        extra={"prop_collection": "PROP_bowl", "scale_reference": "tabletop"},
    ),
    CatalogAsset(
        id="table",
        asset_type="prop",
        blend_file="table.blend",
        notes="Rough hut table sized for family dialogue and meal beats.",
        extra={"prop_collection": "PROP_table", "scale_reference": "adult-waist-high"},
    ),
    CatalogAsset(
        id="bed",
        asset_type="prop",
        blend_file="bed.blend",
        notes="Straw bed prop for hut and exhaustion staging.",
        extra={"prop_collection": "PROP_bed", "scale_reference": "child-length"},
    ),
    CatalogAsset(
        id="firewood",
        asset_type="prop",
        blend_file="firewood.blend",
        notes="Stackable logs, sticks, and kindling for hut and forest beats.",
        extra={"prop_collection": "PROP_firewood", "scale_reference": "arm-carried"},
    ),
    CatalogAsset(
        id="axe",
        asset_type="prop",
        blend_file="axe.blend",
        notes="Woodcutter axe prop for father identity and forest staging.",
        extra={"prop_collection": "PROP_axe", "scale_reference": "adult-hand-held"},
    ),
    CatalogAsset(
        id="basket",
        asset_type="prop",
        blend_file="basket.blend",
        notes="Basket prop for food, stones, and carried objects.",
        extra={"prop_collection": "PROP_basket", "scale_reference": "child-carried"},
    ),
    CatalogAsset(
        id="cage",
        asset_type="prop",
        blend_file="cage.blend",
        notes="Holding cage prop for crone captivity beats.",
        extra={"prop_collection": "PROP_cage", "scale_reference": "child-sized"},
    ),
    CatalogAsset(
        id="oven_door",
        asset_type="prop",
        blend_file="oven_door.blend",
        notes="Animated oven door prop with hinge and latch controls.",
        extra={"prop_collection": "PROP_oven_door", "scale_reference": "adult-height"},
    ),
    CatalogAsset(
        id="jewels",
        asset_type="prop",
        blend_file="jewels.blend",
        notes="Treasure prop set for ending and return beats.",
        extra={"prop_collection": "PROP_jewels", "scale_reference": "handful"},
    ),
    CatalogAsset(
        id="duck",
        asset_type="prop",
        blend_file="duck.blend",
        notes="Reusable white duck model or rig for river crossing.",
        extra={"prop_collection": "PROP_duck", "scale_reference": "small-animal"},
    ),
    CatalogAsset(
        id="door_latches",
        asset_type="prop",
        blend_file="door_latches.blend",
        notes="Reusable latch and hinge pieces for hut, cage, and house doors.",
        extra={"prop_collection": "PROP_door_latches", "scale_reference": "door-hardware"},
    ),
    CatalogAsset(
        id="edible_house_pieces",
        asset_type="prop",
        blend_file="edible_house_pieces.blend",
        notes="Reusable bread, cake, and sugar architectural pieces for the house.",
        extra={"prop_collection": "PROP_edible_house_pieces", "scale_reference": "set-dressing"},
    ),
)


def write_catalog_manifests(root: Path, *, force: bool = False) -> list[Path]:
    written: list[Path] = []
    for asset in PRODUCTION_CATALOG:
        manifest_path = root / asset.manifest_dir / "manifest.yaml"
        if manifest_path.exists() and not force:
            continue
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            yaml.safe_dump(asset.to_manifest(), sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        written.append(manifest_path)
    return written
