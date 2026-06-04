"""Headless asset-build helpers."""

from blenderservices.build.scene import (
    fresh_scene,
    make_collection,
    parse_out_arg,
)
from blenderservices.build.mesh import (
    add_cone,
    add_cube,
    add_cylinder,
    add_mesh_from_verts,
    add_sphere,
    join,
)
from blenderservices.build.armature import (
    BoneSpec,
    bind_mesh_to_armature,
    build_armature,
)
from blenderservices.build.palette import (
    GRIMDARK_COLOURS,
    assign_material,
    build_flat_material,
)
from blenderservices.build.render import (
    FACIAL_SHAPE_KEYS,
    add_shape_keys,
    apply_grimdark_render_settings,
    enable_freestyle_ink,
)
from blenderservices.build.save import (
    log_step,
    normalize_library_paths,
    repo_root,
    resolve_blend_path,
    save_as,
)

__all__ = [
    # scene
    "fresh_scene",
    "make_collection",
    "parse_out_arg",
    # mesh
    "add_cone",
    "add_cube",
    "add_cylinder",
    "add_mesh_from_verts",
    "add_sphere",
    "join",
    # armature
    "BoneSpec",
    "bind_mesh_to_armature",
    "build_armature",
    # palette
    "GRIMDARK_COLOURS",
    "assign_material",
    "build_flat_material",
    # render
    "FACIAL_SHAPE_KEYS",
    "add_shape_keys",
    "apply_grimdark_render_settings",
    "enable_freestyle_ink",
    # save
    "log_step",
    "normalize_library_paths",
    "repo_root",
    "resolve_blend_path",
    "save_as",
]
