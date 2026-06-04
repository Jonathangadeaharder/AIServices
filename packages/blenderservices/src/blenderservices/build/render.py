"""Render settings, Freestyle ink pass, and shape-key helpers."""

from __future__ import annotations

from typing import Iterable

import bpy

from blenderservices.build.palette import GRIMDARK_COLOURS
from blenderservices.episode_settings import (
    DEFAULT_RENDER_FPS,
    DEFAULT_RENDER_HEIGHT,
    DEFAULT_RENDER_WIDTH,
)


# The 12 universal facial shape keys — identical names across every character.
# Animators learn one control set and it drives any actor.
FACIAL_SHAPE_KEYS = [
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
]


def add_shape_keys(obj, names: Iterable[str]) -> None:
    """Add a Basis key (if missing) and one shape key per name. Values default to 0."""
    if obj.type != "MESH":
        raise ValueError(f"{obj.name} is not a mesh; can't add shape keys")
    if not obj.data.shape_keys:
        obj.shape_key_add(name="Basis", from_mix=False)
    existing = {k.name for k in obj.data.shape_keys.key_blocks}
    for n in names:
        if n not in existing:
            obj.shape_key_add(name=n, from_mix=False)


def enable_freestyle_ink(thickness: float = 2.4) -> None:
    """Enable freestyle on the active scene with the GrimDark settings."""
    scene = bpy.context.scene
    scene.render.use_freestyle = True

    vl = scene.view_layers[0]
    vl.use_freestyle = True

    # default line set: silhouette + crease + edge mark + material boundary
    fs = vl.freestyle_settings
    if not fs.linesets:
        fs.linesets.new("grimdark_ink")
    while len(fs.linesets) > 1:
        fs.linesets.remove(fs.linesets[-1])
    ls = fs.linesets[0]
    ls.name = "grimdark_ink"
    ls.select_silhouette = True
    ls.select_crease = True
    ls.select_border = True
    ls.select_edge_mark = True
    ls.select_external_contour = True
    # Exclude face-marked geometry from ALL freestyle lines. Dense displaced meshes (e.g.
    # carved spiky hair) otherwise get every facet edge inked into a grey scribble. Marking
    # those meshes' faces drops them here. Harmless where nothing is marked (nothing excluded).
    if hasattr(ls, "select_by_face_marks"):
        ls.select_by_face_marks = True
        ls.face_mark_condition = "ONE"
        ls.face_mark_negation = "EXCLUSIVE"

    style = ls.linestyle
    if style is None:
        style = bpy.data.linestyles.new("grimdark_ink_style")
        ls.linestyle = style
    style.color = GRIMDARK_COLOURS["ink"][:3]
    style.thickness = thickness
    style.thickness_position = "CENTER"

    # add a sketchy / chained-line modifier so the stroke isn't dead-straight
    if "grimdark_sketch" not in [m.name for m in style.geometry_modifiers]:
        m = style.geometry_modifiers.new(name="grimdark_sketch", type="SINUS_DISPLACEMENT")
        m.amplitude = 0.3
        m.wavelength = 12.0


def apply_grimdark_render_settings() -> None:
    """The render preset every shot starts from. See docs/00_pipeline.md."""
    scene = bpy.context.scene
    # EEVEE Next is the only engine
    scene.render.engine = (
        "BLENDER_EEVEE_NEXT"
        if hasattr(bpy.app, "version") and bpy.app.version >= (4, 2, 0)
        else "BLENDER_EEVEE"
    )

    # 2.4:1 cinemascope letterbox
    scene.render.resolution_x = DEFAULT_RENDER_WIDTH
    scene.render.resolution_y = DEFAULT_RENDER_HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.fps = DEFAULT_RENDER_FPS

    # AgX (fall back to Standard, then Filmic) — Filmic's long desaturating shoulder
    # gamut-compresses the warm amber carved-wood albedo to pale cream/grey. AgX keeps
    # saturated warm tones to screen, which is the carved-puppet plate look.
    for _vt in ("AgX", "Standard", "Filmic"):
        try:
            scene.view_settings.view_transform = _vt
            break
        except TypeError:
            continue
    enable_freestyle_ink()

    # world ambient is solid ink (no HDRI, ever)
    world = scene.world or bpy.data.worlds.new("grimdark_dark")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = GRIMDARK_COLOURS["ink"]
    bg.inputs["Strength"].default_value = 0.6
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
