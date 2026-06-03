"""Shared rig lookup helpers for Blender shot scripts."""

from __future__ import annotations

RIG_BY_CHARACTER: dict[str, str] = {
    "gretel": "gretel_rig",
    "hansel": "hansel_rig",
    "crone": "crone_rig",
    "father": "father_rig",
    "stepmother": "stepmother_rig",
}

MIXAMO_PREFIXES = tuple(
    f"mixamorig{suffix}:" for suffix in ("", "_", "1", "2", "3", "4", "11", "12", "21", "22")
)


def _get_named(collection, name: str):
    if hasattr(collection, "get"):
        return collection.get(name)
    try:
        return collection[name]
    except KeyError:
        return None


def _flatten_names(names):
    flat = []
    for name in names:
        if isinstance(name, (tuple, list)):
            flat.extend(name)
        else:
            flat.append(name)
    return tuple(str(name) for name in flat if name)


def resolve_pose_bone(pose_bones, *names: str):
    """Resolve a pose bone by exact name, known Mixamo prefix, then namespace suffix."""
    names = _flatten_names(names)
    for name in names:
        bone = _get_named(pose_bones, name)
        if bone is not None:
            return bone
        for prefix in MIXAMO_PREFIXES:
            bone = _get_named(pose_bones, f"{prefix}{name}")
            if bone is not None:
                return bone
    for name in names:
        suffix = f":{name}"
        for bone in pose_bones:
            if bone.name.endswith(suffix):
                return bone
    return None


def resolve_bone(bones, logical: str):
    """Resolve an edit/data bone with the same exact -> prefix -> namespace suffix policy."""
    logical = _flatten_names((logical,))[0] if logical else logical
    bone = _get_named(bones, logical)
    if bone is not None:
        return bone
    for prefix in MIXAMO_PREFIXES:
        bone = _get_named(bones, f"{prefix}{logical}")
        if bone is not None:
            return bone
    suffix = f":{logical}"
    for bone in bones:
        if bone.name.endswith(suffix):
            return bone
    return None


def resolve_bone_name(bones, logical: str) -> str | None:
    bone = resolve_bone(bones, logical)
    return bone.name if bone is not None else None


def resolve_vertex_group(vertex_groups, logical: str) -> int | None:
    group = _get_named(vertex_groups, logical)
    if group is not None:
        return group.index
    for prefix in MIXAMO_PREFIXES:
        group = _get_named(vertex_groups, f"{prefix}{logical}")
        if group is not None:
            return group.index
    suffix = f":{logical}"
    for group in vertex_groups:
        if group.name.endswith(suffix):
            return group.index
    return None


def find_armature(char_id: str):
    """Find the armature for a character in linked shot collections or by object name."""
    import bpy

    preferred = RIG_BY_CHARACTER.get(char_id, f"{char_id}_rig")
    obj = bpy.data.objects.get(preferred)
    if obj is not None and obj.type == "ARMATURE":
        return obj

    chars = bpy.data.collections.get("10_CHARS")
    if chars is not None:
        for child in chars.children:
            if child.name == char_id or child.name.lower() == char_id.lower():
                for item in child.all_objects:
                    if item.type == "ARMATURE":
                        return item

    lowered = char_id.lower()
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and lowered in obj.name.lower():
            return obj
    return None
