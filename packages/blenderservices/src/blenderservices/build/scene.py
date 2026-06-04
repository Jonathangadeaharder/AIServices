"""Scene cleanup, argument parsing, and collection helpers."""

from __future__ import annotations

import argparse

import bpy

from blenderservices.bpy_common import blender_argv


def parse_out_arg(default: str | None = None) -> str | None:
    """Pull --out from argv after the `--` separator that Blender passes."""
    argv = blender_argv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=default)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    return args.out or args.out_dir


def fresh_scene() -> None:
    """Wipe the default scene. Always start from zero."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Set unit system to metric, scale 1.0 (default).
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0


def make_collection(name: str, parent: bpy.types.Collection | None = None) -> bpy.types.Collection:
    """Idempotently create a named collection under `parent` (or scene root)."""
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    coll = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(coll)
    return coll


def _move_to_collection(obj, collection_name: str | None) -> None:
    if not collection_name:
        return
    target = make_collection(collection_name)
    # remove from any existing collections
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    target.objects.link(obj)
