"""Headless asset-build helpers.

Import specific modules directly (e.g. ``from blenderservices.build.scene import fresh_scene``).
These submodules require ``bpy``, so they can only be imported inside Blender.
"""

__all__ = [
    "fresh_scene",
    "make_collection",
    "parse_out_arg",
    "add_cone",
    "add_cube",
    "add_cylinder",
    "add_mesh_from_verts",
    "add_sphere",
    "join",
    "BoneSpec",
    "bind_mesh_to_armature",
    "build_armature",
    "GRIMDARK_COLOURS",
    "assign_material",
    "build_flat_material",
    "FACIAL_SHAPE_KEYS",
    "add_shape_keys",
    "apply_grimdark_render_settings",
    "enable_freestyle_ink",
    "log_step",
    "normalize_library_paths",
    "project_root",
    "resolve_blend_path",
    "save_as",
]
