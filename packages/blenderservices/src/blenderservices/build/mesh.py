"""Primitive mesh builders and join utility."""

from __future__ import annotations

from typing import Sequence

import bpy

from blenderservices.build.scene import _move_to_collection


def add_cube(name: str, location, scale, collection: str | None = None):
    """Add an axis-aligned cube, scaled to `scale` (half-extents)."""
    bpy.ops.mesh.primitive_cube_add(size=2, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    _move_to_collection(obj, collection)
    return obj


def add_sphere(
    name: str, location, radius=1.0, segments=16, rings=8, collection: str | None = None
):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments, ring_count=rings, radius=radius, location=location
    )
    obj = bpy.context.active_object
    obj.name = name
    _move_to_collection(obj, collection)
    return obj


def add_cylinder(
    name: str, location, radius=0.1, depth=1.0, vertices=12, collection: str | None = None
):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices, radius=radius, depth=depth, location=location
    )
    obj = bpy.context.active_object
    obj.name = name
    _move_to_collection(obj, collection)
    return obj


def add_cone(
    name: str, location, radius=0.3, depth=0.6, vertices=12, collection: str | None = None
):
    bpy.ops.mesh.primitive_cone_add(
        vertices=vertices,
        radius1=radius,
        radius2=0.0,
        depth=depth,
        location=location,
    )
    obj = bpy.context.active_object
    obj.name = name
    _move_to_collection(obj, collection)
    return obj


def add_mesh_from_verts(
    name: str, verts: Sequence, faces: Sequence, location=(0, 0, 0), collection: str | None = None
):
    """Build an arbitrary mesh from a vert/face list."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(list(verts), [], list(faces))
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    _move_to_collection(obj, collection)
    return obj


def join(target_name: str, *objects, collection: str | None = None):
    """Join multiple meshes into one. First object survives."""
    bpy.ops.object.select_all(action="DESELECT")
    base = objects[0]
    base.name = target_name
    base.select_set(True)
    bpy.context.view_layer.objects.active = base
    for obj in objects[1:]:
        obj.select_set(True)
    bpy.ops.object.join()
    if collection:
        _move_to_collection(base, collection)
    return base
