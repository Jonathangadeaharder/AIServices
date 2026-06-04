"""Carved-wood and grimdark material builders.

Import specific modules directly
(e.g. ``from blenderservices.materials.grimdark import build_carved_wood``).
These submodules require ``bpy``, so they can only be imported inside Blender.
"""

__all__ = [
    "apply_grimdark_view",
    "assign_wood_to_meshes",
    "assign_wood_to_set_meshes",
    "build_carved_wood",
    "build_frosted_glass",
    "build_stone",
    "enable_set_microdisplacement",
    "set_active_mood",
]
