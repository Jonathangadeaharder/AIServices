"""Armature builder and mesh-armature binding."""

from __future__ import annotations

from typing import Iterable

import bpy
from mathutils import Vector

from blenderservices.build.scene import _move_to_collection


class BoneSpec:
    """One bone definition. head, tail in world space. parent by name."""

    __slots__ = ("name", "head", "tail", "parent", "use_connect")

    def __init__(self, name: str, head, tail, parent: str | None = None, use_connect: bool = False):
        self.name = name
        self.head = Vector(head)
        self.tail = Vector(tail)
        self.parent = parent
        self.use_connect = use_connect


def build_armature(
    name: str, bones: Iterable[BoneSpec], collection: str | None = None
) -> bpy.types.Object:
    """Build an armature object from a flat list of BoneSpec."""
    arm_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    _move_to_collection(arm_obj, collection)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        eb = arm_obj.data.edit_bones
        # first pass: create
        for spec in bones:
            b = eb.new(spec.name)
            b.head = spec.head
            b.tail = spec.tail
        # second pass: parents (all bones now exist)
        for spec in bones:
            if spec.parent:
                eb[spec.name].parent = eb[spec.parent]
                eb[spec.name].use_connect = spec.use_connect
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    return arm_obj


def bind_mesh_to_armature(mesh_obj, arm_obj, with_weights: bool = True) -> None:
    """Parent mesh to armature with automatic weights (Blender's heat solver)."""
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type="ARMATURE_AUTO" if with_weights else "ARMATURE")
