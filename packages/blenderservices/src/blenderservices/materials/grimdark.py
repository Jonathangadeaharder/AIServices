"""Shared carved-wood material assignment for Mixamo and look-dev probes."""

from __future__ import annotations

import math

import bpy

from blenderservices.build.palette import GRIMDARK_COLOURS
from blenderservices.build.render import apply_grimdark_render_settings

_MOOD_TINTS: dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
] = {
    # (shadow_tint, mid_tint, hi_tint) multipliers applied to base color.
    "moon": ((0.22, 0.26, 0.42), (0.55, 0.62, 0.85), (0.95, 1.00, 1.10)),
    "ember": ((0.30, 0.18, 0.12), (0.85, 0.55, 0.32), (1.20, 0.95, 0.70)),
    "noon": ((0.32, 0.32, 0.32), (0.72, 0.72, 0.72), (1.05, 1.02, 0.95)),
    "neutral": ((0.40, 0.40, 0.40), (0.78, 0.78, 0.78), (1.00, 1.00, 1.00)),
}

_ACTIVE_MOOD = "neutral"


def set_active_mood(mood: str) -> None:
    """Set the compositor mood tint used by render probes."""
    global _ACTIVE_MOOD
    _ACTIVE_MOOD = mood if mood in _MOOD_TINTS else "neutral"


def _apply_mood(rgb: tuple[float, float, float], band: str) -> tuple[float, float, float, float]:
    s, m, h = _MOOD_TINTS[_ACTIVE_MOOD]
    mult = {"shadow": s, "mid": m, "hi": h}[band]
    out = tuple(max(0.0, min(1.0, c * k)) for c, k in zip(rgb, mult, strict=False))
    return (*out, 1.0)


def _pick_slot(mesh_name: str) -> str:
    nm = mesh_name.lower()
    if any(k in nm for k in ("eyelash", "eyebrow")):
        return "hair"
    if any(k in nm for k in ("eye", "iris", "pupil")):
        return "eye"
    if any(k in nm for k in ("hair", "beard")):
        return "hair"
    if any(k in nm for k in ("sneaker", "shoe", "boot", "sock")):
        return "shoe"
    if any(
        k in nm
        for k in (
            "cloth",
            "shirt",
            "pant",
            "trouser",
            "dress",
            "apron",
            "vest",
            "jacket",
            "coat",
            "top",
            "bottom",
            "short",
            "skirt",
            "hat",
            "cap",
            "helmet",
            "hard",
        )
    ):
        return "cloth"
    return "skin"


def _shade_for_ink(obj: bpy.types.Object) -> None:
    """Smooth surfaces with sharp creases so carved silhouettes stay readable."""
    mesh = obj.data
    # If the object is chiseled (has CarveDecimate modifier), keep flat shading!
    has_carve_dec = "CarveDecimate" in obj.modifiers
    if has_carve_dec:
        for poly in mesh.polygons:
            poly.use_smooth = False
        # Remove any EDGE_SPLIT modifiers added by previous steps
        for mod in list(obj.modifiers):
            if mod.name == "grimdark_crease":
                obj.modifiers.remove(mod)
    else:
        for poly in mesh.polygons:
            poly.use_smooth = True
        if not any(mod.type == "EDGE_SPLIT" for mod in obj.modifiers):
            mod = obj.modifiers.new("grimdark_crease", "EDGE_SPLIT")
            mod.split_angle = 0.65  # ~37°
        if hasattr(mesh, "use_auto_smooth"):
            mesh.use_auto_smooth = True
            mesh.auto_smooth_angle = 0.65


_SET_TONES: dict[str, tuple[float, float, float]] = {
    # Differentiate set surfaces so walls/floor/hearth/bed don't all read as
    # the same gray-brown mass. Tones tuned for engraved-woodcut feel.
    "wall": (0.22, 0.20, 0.18),
    "floor": (0.14, 0.12, 0.10),
    "ceiling": (0.10, 0.09, 0.08),
    "hearth": (0.30, 0.18, 0.12),
    "crate": (
        0.46,
        0.30,
        0.15,
    ),  # warm wood; without this "crate" fell through to the cool grey wall tone
    "bed": (0.42, 0.34, 0.22),
    "table": (0.32, 0.24, 0.16),
    "stool": (0.32, 0.24, 0.16),
    "bench": (0.32, 0.24, 0.16),
    "roof": (0.18, 0.13, 0.10),
    "ground": (0.16, 0.13, 0.09),
}


def _set_tone_for(name: str) -> tuple[float, float, float]:
    nm = name.lower()
    for key, rgb in _SET_TONES.items():
        if key in nm:
            return rgb
    return _SET_TONES["wall"]


def _is_set_mesh(name: str) -> bool:
    nm = name.lower()
    if any(k in nm for k in ("ch02", "ch17", "bryce", "char_", "axe", "prop_")):
        return False
    return any(
        k in nm
        for k in (
            "hut_",
            "set_",
            "table",
            "wall",
            "floor",
            "roof",
            "bed_",
            "hearth",
            "ground",
            "stool",
            "bench",
            "fireplace",
            "beam",
            "crate",
        )
    )


# =============================================================================
# Carved-wood material (Cycles-safe) — matches style_targets carved-puppet plates
# =============================================================================
# Warm pine/oak central tone per slot. The grain ramp spans darker->lighter around it.
_WOOD_TONES: dict[str, tuple[float, float, float]] = {
    # Darker + warmer/more-saturated than the old pale-tan tones: the target wood is rich
    # amber/orange, not pine-cream. Lower value + wider red:blue gap survives the AgX
    # tonemap as saturated low-key wood instead of washing to grey.
    "skin": (0.56, 0.34, 0.17),
    "cloth": (0.42, 0.28, 0.15),
    "hair": (0.28, 0.17, 0.09),
    "eye": (0.14, 0.09, 0.05),
    "shoe": (0.24, 0.15, 0.08),
}
# Per-character grain tone shift (keeps silhouettes distinct, all woody).
_WOOD_CHAR_TINT: dict[str, dict[str, tuple[float, float, float]]] = {
    "gretel": {"skin": (0.62, 0.40, 0.21), "hair": (0.40, 0.24, 0.11)},
    "hansel": {"skin": (0.60, 0.37, 0.19), "hair": (0.28, 0.16, 0.08)},
    # father beard/hair: weathered MID grey under the new low-key exposure — not a light
    # grey that becomes the brightest object in a dark frame.
    "father": {"skin": (0.50, 0.33, 0.19), "hair": (0.50, 0.47, 0.43)},
    "stepmother": {"skin": (0.78, 0.60, 0.42), "hair": (0.20, 0.13, 0.12)},
    "crone": {"skin": (0.62, 0.50, 0.36), "hair": (0.74, 0.70, 0.62)},
}


def _char_from_collections(obj: bpy.types.Object) -> str | None:
    for coll in obj.users_collection:
        n = coll.name.lower()
        for char in _WOOD_CHAR_TINT:
            if char in n:
                return char
    return None


# Painted-tunic/dress accents washed over the bare wood (grain reads through the
# MULTIPLY). Plate-faithful: hansel green tunic, gretel umber dress, father olive
# vest, crone black, stepmother dusky red.
_WOOD_PAINT: dict[str, dict[str, tuple[float, float, float]]] = {
    "hansel": {
        "cloth": (0.20, 0.34, 0.16)
    },  # mossy desaturated green over wood grain (not lime knit)
    "gretel": {"cloth": (0.58, 0.26, 0.18)},
    "father": {"cloth": (0.40, 0.46, 0.20)},
    "stepmother": {"cloth": (0.50, 0.16, 0.22)},
    "crone": {"cloth": (0.10, 0.09, 0.11)},
}

# Per-slot grain direction. Body forms (skin/cloth) get vertical grain along the
# limb/torso; planks/sets keep the default Z stacking (handled in their own builder).
_GRAIN_ROT: dict[str, tuple[float, float, float]] = {
    "skin": (0.0, math.pi * 0.5, 0.0),
    "cloth": (0.0, math.pi * 0.5, 0.0),
}


def _hatch_mask(nt: bpy.types.NodeTree, coords, scale: float, rot_z: float):
    """0..1 value, dipping to ~0 along thin parallel engraving lines."""
    m = nt.nodes.new("ShaderNodeMapping")
    m.inputs["Rotation"].default_value = (0.0, 0.0, rot_z)
    nt.links.new(coords, m.inputs["Vector"])
    w = nt.nodes.new("ShaderNodeTexWave")
    w.wave_type = "BANDS"
    w.bands_direction = "X"
    w.wave_profile = "SIN"
    w.inputs["Scale"].default_value = scale
    w.inputs["Distortion"].default_value = 1.0
    w.inputs["Detail"].default_value = 1.0
    nt.links.new(m.outputs["Vector"], w.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "B_SPLINE"
    e = ramp.color_ramp.elements
    e[0].position = 0.0
    e[0].color = (0.0, 0.0, 0.0, 1.0)
    e[1].position = 0.28  # narrow dark line, wide light ground
    e[1].color = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(w.outputs["Fac"], ramp.inputs["Fac"])
    bw = nt.nodes.new("ShaderNodeRGBToBW")
    nt.links.new(ramp.outputs["Color"], bw.inputs["Color"])
    return bw.outputs["Val"]


def _grain_color(
    nt: bpy.types.NodeTree,
    coords,
    base_rgb: tuple[float, float, float],
    *,
    grain_type: str,
    grain_scale: float,
    grain_contrast: float,
    grain_warp: float,
    grain_rot: tuple[float, float, float],
    knots: bool,
):
    """Build the warm wood-grain albedo. Returns (color_socket, grain_fac_socket).

    The warp is two-scale: a fine TexNoise wobble plus a low-freq TexNoise so plank
    bands meander per-region instead of running as regular zebra stripes. ``grain_rot``
    rotates the grain coords so direction follows the form (limbs/torso vertical, planks
    horizontal). ``knots`` scatters sparse dark elliptical knots into the ramp colour.
    """
    # Per-slot grain direction (rotate the coords that feed the wave + warp).
    if any(grain_rot):
        gmap = nt.nodes.new("ShaderNodeMapping")
        gmap.inputs["Rotation"].default_value = grain_rot
        nt.links.new(coords, gmap.inputs["Vector"])
        gco = gmap.outputs["Vector"]
    else:
        gco = coords

    # Fine wobble.
    distort = nt.nodes.new("ShaderNodeTexNoise")
    distort.inputs["Scale"].default_value = 2.5
    distort.inputs["Detail"].default_value = 2.0
    nt.links.new(gco, distort.inputs["Vector"])
    warp_hi = nt.nodes.new("ShaderNodeVectorMath")
    warp_hi.operation = "SCALE"
    warp_hi.inputs["Scale"].default_value = grain_warp
    nt.links.new(distort.outputs["Color"], warp_hi.inputs[0])
    # Low-freq regional meander so bands wander per-plank, not as regular bands.
    meander = nt.nodes.new("ShaderNodeTexNoise")
    meander.inputs["Scale"].default_value = 0.8
    meander.inputs["Detail"].default_value = 1.0
    nt.links.new(gco, meander.inputs["Vector"])
    warp_lo = nt.nodes.new("ShaderNodeVectorMath")
    warp_lo.operation = "SCALE"
    warp_lo.inputs["Scale"].default_value = grain_warp * 1.6
    nt.links.new(meander.outputs["Color"], warp_lo.inputs[0])
    warp_sum = nt.nodes.new("ShaderNodeVectorMath")
    warp_sum.operation = "ADD"
    nt.links.new(warp_hi.outputs["Vector"], warp_sum.inputs[0])
    nt.links.new(warp_lo.outputs["Vector"], warp_sum.inputs[1])
    warp = nt.nodes.new("ShaderNodeVectorMath")
    warp.operation = "ADD"
    nt.links.new(gco, warp.inputs[0])
    nt.links.new(warp_sum.outputs["Vector"], warp.inputs[1])

    grain = nt.nodes.new("ShaderNodeTexWave")
    grain.wave_type = "RINGS" if grain_type == "rings" else "BANDS"
    grain.bands_direction = "Z"
    grain.wave_profile = "SIN"
    grain.inputs["Scale"].default_value = grain_scale
    # Lower distortion + higher detail-roughness wanders the band CENTRES (kills the
    # regular stripe rhythm) instead of only jittering band EDGES at high frequency.
    grain.inputs["Distortion"].default_value = 1.4
    grain.inputs["Detail"].default_value = 2.5
    if "Detail Roughness" in grain.inputs:
        grain.inputs["Detail Roughness"].default_value = 0.85
    nt.links.new(warp.outputs["Vector"], grain.inputs["Vector"])
    gramp = nt.nodes.new("ShaderNodeValToRGB")
    ge = gramp.color_ramp.elements
    lo = 1.0 - 0.55 * grain_contrast
    hi = 1.0 + 0.30 * grain_contrast
    ge[0].position = 0.0
    ge[0].color = (*tuple(c * lo for c in base_rgb), 1.0)
    ge[1].position = 1.0
    ge[1].color = (*tuple(min(1.0, c * hi) for c in base_rgb), 1.0)
    gmid = gramp.color_ramp.elements.new(0.5)
    gmid.color = (*base_rgb, 1.0)
    nt.links.new(grain.outputs["Fac"], gramp.inputs["Fac"])
    grain_col = gramp.outputs["Color"]

    # Sparse dark elliptical knots — F1 voronoi thresholded near its feature points.
    if knots:
        kvor = nt.nodes.new("ShaderNodeTexVoronoi")
        kvor.feature = "F1"
        kvor.inputs["Scale"].default_value = 3.0
        nt.links.new(gco, kvor.inputs["Vector"])
        kmap = nt.nodes.new("ShaderNodeMapRange")
        kmap.inputs["From Min"].default_value = 0.0
        kmap.inputs["From Max"].default_value = 0.12  # only the cell core darkens
        kmap.inputs["To Min"].default_value = 1.0
        kmap.inputs["To Max"].default_value = 0.0
        kmap.clamp = True
        nt.links.new(kvor.outputs["Distance"], kmap.inputs["Value"])
        kmix = nt.nodes.new("ShaderNodeMixRGB")
        kmix.blend_type = "MIX"
        kmix.inputs["Color2"].default_value = (
            *tuple(c * 0.28 for c in base_rgb),
            1.0,
        )
        nt.links.new(kmap.outputs["Result"], kmix.inputs["Fac"])
        nt.links.new(grain_col, kmix.inputs["Color1"])
        grain_col = kmix.outputs["Color"]

    return grain_col, grain.outputs["Fac"]


def build_carved_wood(
    name: str,
    base_rgb: tuple[float, float, float],
    *,
    grain_type: str = "bands",
    grain_scale: float = 6.0,
    grain_contrast: float = 0.55,
    grain_warp: float = 0.10,
    grain_rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    knots: bool = False,
    hatch_scale: float = 22.0,
    crosshatch: bool = True,
    fine_hatch: bool = False,
    engrave: bool = True,
    paint_rgb: tuple[float, float, float] | None = None,
    paint_fac: float = 0.6,
    paint_wear: bool = True,
    bump_hatch_strength: float = 0.13,
    bump_hatch_distance: float = 0.008,
    face_bump: bool = False,
) -> bpy.types.Material:
    """Cycles carved-wood: procedural grain + engraved cross-hatch grooves. The hatch
    is physical (bump + slight albedo darkening) so real chiaroscuro lighting shades it
    into woodcut shadow lines without an EEVEE-only screen-space quantization pass.

    Plate-fidelity controls: ``grain_warp``/``grain_rot``/``knots`` shape the grain;
    ``paint_rgb`` washes a flat paint colour over the wood (grain reads through) with
    ``paint_wear`` letting bare wood show on convex edges; ``fine_hatch`` adds a second
    finer cross-hatch layer for dense clothing shadow.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.use_backface_culling = False
    nt = mat.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.86
    for spec_name in ("Specular IOR Level", "Specular"):
        if spec_name in bsdf.inputs:
            bsdf.inputs[spec_name].default_value = 0.01
            break
    # Warm subsurface so lit carved wood glows amber under the key (the target plates read
    # as translucent oiled wood, not matte plastic). Cheap — no displacement. Radius skewed
    # red/amber; tiny scale so only thin edges/rim transmit.
    for w_name in ("Subsurface Weight", "Subsurface"):
        if w_name in bsdf.inputs:
            bsdf.inputs[w_name].default_value = 0.12
            break
    if "Subsurface Radius" in bsdf.inputs:
        bsdf.inputs["Subsurface Radius"].default_value = (0.62, 0.26, 0.12)
    if "Subsurface Scale" in bsdf.inputs:
        bsdf.inputs["Subsurface Scale"].default_value = 0.06
    co = nt.nodes.new("ShaderNodeTexCoord")
    coords = co.outputs["Object"]

    grain_col, grain_fac = _grain_color(
        nt,
        coords,
        base_rgb,
        grain_type=grain_type,
        grain_scale=grain_scale,
        grain_contrast=grain_contrast,
        grain_warp=grain_warp,
        grain_rot=grain_rot,
        knots=knots,
    )

    # --- painted colour accents over wood. The paint hue is the base; the wood grain
    #     reads through as value (light/dark) variation, and bare wood is worn through
    #     on convex edges/corners. A flat MULTIPLY muddies mid paint over mid wood, so
    #     instead we modulate the paint's brightness by the grain luminance. ---
    if paint_rgb is not None:
        g_bw = nt.nodes.new("ShaderNodeRGBToBW")
        nt.links.new(grain_col, g_bw.inputs["Color"])
        gmod = nt.nodes.new("ShaderNodeMapRange")
        gmod.inputs["From Min"].default_value = 0.12
        gmod.inputs["From Max"].default_value = 0.70
        gmod.inputs["To Min"].default_value = 1.0 - 0.45 * paint_fac
        gmod.inputs["To Max"].default_value = 1.0 + 0.18 * paint_fac
        gmod.clamp = True
        nt.links.new(g_bw.outputs["Val"], gmod.inputs["Value"])
        painted = nt.nodes.new("ShaderNodeVectorMath")
        painted.operation = "SCALE"
        painted.inputs[0].default_value = (paint_rgb[0], paint_rgb[1], paint_rgb[2])
        nt.links.new(gmod.outputs["Result"], painted.inputs["Scale"])
        painted_col = painted.outputs["Vector"]
        if paint_wear:
            geo = nt.nodes.new("ShaderNodeNewGeometry")
            wear = nt.nodes.new("ShaderNodeValToRGB")
            we = wear.color_ramp.elements
            we[0].position = 0.50
            we[0].color = (0.0, 0.0, 0.0, 1.0)  # flat faces -> keep paint
            we[1].position = 0.62
            we[1].color = (1.0, 1.0, 1.0, 1.0)  # convex edges -> bare wood worn through
            nt.links.new(geo.outputs["Pointiness"], wear.inputs["Fac"])
            wmix = nt.nodes.new("ShaderNodeMixRGB")
            wmix.blend_type = "MIX"
            nt.links.new(wear.outputs["Color"], wmix.inputs["Fac"])
            nt.links.new(painted_col, wmix.inputs["Color1"])
            nt.links.new(grain_col, wmix.inputs["Color2"])
            grain_col = wmix.outputs["Color"]
        else:
            grain_col = painted_col

    # --- engraved cross-hatch (skippable: hair/round forms read as a grid-net) ---
    if not engrave:
        nt.links.new(grain_col, bsdf.inputs["Base Color"])
        voro0 = nt.nodes.new("ShaderNodeTexVoronoi")
        voro0.inputs["Scale"].default_value = 14.0
        nt.links.new(coords, voro0.inputs["Vector"])
        bg = nt.nodes.new("ShaderNodeBump")
        bg.inputs["Strength"].default_value = 0.10
        nt.links.new(grain_fac, bg.inputs["Height"])
        bt = nt.nodes.new("ShaderNodeBump")
        bt.inputs["Strength"].default_value = 0.12
        nt.links.new(voro0.outputs["Distance"], bt.inputs["Height"])
        nt.links.new(bg.outputs["Normal"], bt.inputs["Normal"])
        nt.links.new(bt.outputs["Normal"], bsdf.inputs["Normal"])
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        ink0 = GRIMDARK_COLOURS["ink"]
        mat.line_color = ink0[:4] if len(ink0) > 3 else (*ink0[:3], 1.0)
        mat.line_priority = 1
        return mat
    h1 = _hatch_mask(nt, coords, hatch_scale, 0.6)
    if crosshatch:
        h2 = _hatch_mask(nt, coords, hatch_scale, 0.6 + 1.5708)
        hatch = nt.nodes.new("ShaderNodeMath")
        hatch.operation = "MULTIPLY"  # 0 where either line crosses -> lattice
        nt.links.new(h1, hatch.inputs[0])
        nt.links.new(h2, hatch.inputs[1])
        hatch_val = hatch.outputs["Value"]
    else:
        hatch_val = h1
    # Second, finer hatch layer for dense clothing shadow (plate-like engraving).
    if fine_hatch:
        hf = _hatch_mask(nt, coords, hatch_scale * 2.2, 0.6 + 0.35)
        fine_mul = nt.nodes.new("ShaderNodeMath")
        fine_mul.operation = "MULTIPLY"
        nt.links.new(hatch_val, fine_mul.inputs[0])
        nt.links.new(hf, fine_mul.inputs[1])
        hatch_val = fine_mul.outputs["Value"]

    # albedo darkened along engraving lines (subtle; lighting does the rest)
    darken = nt.nodes.new("ShaderNodeMath")
    darken.operation = "MULTIPLY"
    darken.inputs[1].default_value = 0.30
    nt.links.new(hatch_val, darken.inputs[0])
    line_fac = nt.nodes.new("ShaderNodeMath")
    line_fac.operation = "ADD"
    line_fac.inputs[0].default_value = 0.70  # 0.70 + 0.30*mask -> lines 30% darker
    nt.links.new(darken.outputs["Value"], line_fac.inputs[1])
    col_mul = nt.nodes.new("ShaderNodeVectorMath")
    col_mul.operation = "SCALE"
    nt.links.new(grain_col, col_mul.inputs[0])
    nt.links.new(line_fac.outputs["Value"], col_mul.inputs["Scale"])
    nt.links.new(col_mul.outputs["Vector"], bsdf.inputs["Base Color"])

    # --- relief: grain + engraving grooves + blocky tool-mark voronoi ---
    voro = nt.nodes.new("ShaderNodeTexVoronoi")
    voro.inputs["Scale"].default_value = 16.0
    nt.links.new(coords, voro.inputs["Vector"])
    bump_grain = nt.nodes.new("ShaderNodeBump")
    bump_grain.inputs["Strength"].default_value = 0.07
    bump_grain.inputs["Distance"].default_value = 0.015
    nt.links.new(grain_fac, bump_grain.inputs["Height"])
    bump_hatch = nt.nodes.new("ShaderNodeBump")
    bump_hatch.inputs["Strength"].default_value = bump_hatch_strength
    bump_hatch.inputs["Distance"].default_value = bump_hatch_distance
    nt.links.new(hatch_val, bump_hatch.inputs["Height"])
    nt.links.new(bump_grain.outputs["Normal"], bump_hatch.inputs["Normal"])
    bump_tool = nt.nodes.new("ShaderNodeBump")
    bump_tool.inputs["Strength"].default_value = 0.10
    bump_tool.inputs["Distance"].default_value = 0.015
    nt.links.new(voro.outputs["Distance"], bump_tool.inputs["Height"])
    nt.links.new(bump_hatch.outputs["Normal"], bump_tool.inputs["Normal"])
    last_normal = bump_tool.outputs["Normal"]
    # Shallow large-cell voronoi (scale ~5) reads as carved facial masses — deepens
    # eye sockets / brow / cheek planes so the face is chiselled, not a smooth mannequin.
    if face_bump:
        fvor = nt.nodes.new("ShaderNodeTexVoronoi")
        fvor.feature = "F1"
        fvor.inputs["Scale"].default_value = 5.0
        fvor.inputs["Randomness"].default_value = 0.9
        nt.links.new(coords, fvor.inputs["Vector"])
        bump_face = nt.nodes.new("ShaderNodeBump")
        bump_face.inputs["Strength"].default_value = 0.30
        bump_face.inputs["Distance"].default_value = 0.04
        nt.links.new(fvor.outputs["Distance"], bump_face.inputs["Height"])
        nt.links.new(last_normal, bump_face.inputs["Normal"])
        last_normal = bump_face.outputs["Normal"]
    nt.links.new(last_normal, bsdf.inputs["Normal"])

    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    # Real grain + engraving relief (vs bump-only): grain ridges stand proud, hatch lines
    # cut grooves. With adaptive subdivision the key rakes across carved wood the way the
    # plate's deep grain catches light. Harmless without subdiv (acts as extra bump).
    wdisp = nt.nodes.new("ShaderNodeDisplacement")
    wdisp.inputs["Scale"].default_value = 0.009  # finer grain — carved, not over-striated
    wdisp.inputs["Midlevel"].default_value = 0.5
    groove = nt.nodes.new("ShaderNodeMath")
    groove.operation = "MULTIPLY_ADD"
    groove.inputs[
        1
    ].default_value = (
        0.30  # grain ridges — shallow so the key doesn't emboss bands into relief stripes
    )
    groove.inputs[2].default_value = 0.0
    nt.links.new(grain_fac, groove.inputs[0])
    cut = nt.nodes.new("ShaderNodeMath")
    cut.operation = "MULTIPLY"
    cut.inputs[1].default_value = 0.5  # engraving grooves recess
    nt.links.new(hatch_val, cut.inputs[0])
    add_h = nt.nodes.new("ShaderNodeMath")
    add_h.operation = "ADD"
    nt.links.new(groove.outputs["Value"], add_h.inputs[0])
    nt.links.new(cut.outputs["Value"], add_h.inputs[1])
    nt.links.new(add_h.outputs["Value"], wdisp.inputs["Height"])
    nt.links.new(wdisp.outputs["Displacement"], out.inputs["Displacement"])
    if hasattr(mat, "displacement_method"):
        mat.displacement_method = "BOTH"
    ink = GRIMDARK_COLOURS["ink"]
    mat.line_color = ink[:4] if len(ink) > 3 else (*ink[:3], 1.0)
    mat.line_priority = 1
    return mat


_ORGANIC_KEYS = (
    "body",
    "skin",
    "head",
    "hair",
    "beard",
    "cloth",
    "shirt",
    "pant",
    "short",
    "vest",
    "suit",
    "dress",
    "helmet",
    "cap",
    "boot",
    "shoe",
    "sneaker",
    "heel",
    "hand",
    "eyelash",
)


def _is_organic(name: str) -> bool:
    nm = name.lower()
    return nm.startswith("ch") or any(k in nm for k in _ORGANIC_KEYS)


def enable_set_microdisplacement(dicing: float = 2.0, *, include_chars: bool = True) -> int:
    """Add adaptive-subdivision Subsurf so displacement materials produce real carved
    relief (mortar depth, plank/grain ridges, chiselled faces) instead of bump fakery.
    Cycles EXPERIMENTAL only. Box set meshes get SIMPLE subdiv (stays crisp); organic
    char/prop meshes get CATMULL_CLARK (smooth carved wood). For armature-deformed meshes
    the Subsurf is forced LAST so it dices the deformed result without breaking the rig.
    Skips glow/glass and tiny candle beads."""
    scene = bpy.context.scene
    if scene.render.engine != "CYCLES":
        return 0
    try:
        scene.cycles.feature_set = "EXPERIMENTAL"
    except (AttributeError, TypeError):
        return 0
    n = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        has_arm = any(m.type == "ARMATURE" for m in obj.modifiers)
        if has_arm and not include_chars:
            continue
        if _set_material_kind(obj.name) == "glass" or "candle" in obj.name.lower():
            continue
        organic = _is_organic(obj.name) or has_arm
        if "micro_subd" not in obj.modifiers:
            m = obj.modifiers.new("micro_subd", "SUBSURF")
            # CATMULL_CLARK = smooth carved wood for figures; SIMPLE keeps box walls crisp.
            m.subdivision_type = "CATMULL_CLARK" if organic else "SIMPLE"
            m.levels = 1
            m.render_levels = 2
        if has_arm:
            # Adaptive subdiv must dice the final (deformed) mesh → keep it last.
            with bpy.context.temp_override(object=obj, active_object=obj):
                bpy.ops.object.modifier_move_to_index(
                    modifier="micro_subd", index=len(obj.modifiers) - 1
                )
        if hasattr(obj, "cycles"):
            obj.cycles.use_adaptive_subdivision = True
            obj.cycles.dicing_rate = dicing
        n += 1
    return n


def build_stone(
    name: str = "wood_stone",
    base_rgb: tuple[float, float, float] = (0.24, 0.225, 0.205),
    *,
    is_floor: bool = False,
) -> bpy.types.Material:
    """Chunky carved-stone for hearths/ovens: voronoi cell blocks, grey ramp, deep
    blocky bump so the key rakes across mortared masonry (matches the plate hearths)."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.use_backface_culling = False
    nt = mat.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.95
    for spec_name in ("Specular IOR Level", "Specular"):
        if spec_name in bsdf.inputs:
            bsdf.inputs[spec_name].default_value = 0.05
            break
    co = nt.nodes.new("ShaderNodeTexCoord")
    coords = co.outputs["Object"]
    if is_floor:
        brick_coords = coords
    else:
        # Brick patterns in the X-Y plane, but the dominant stone surfaces here are vertical
        # walls (X-Z face, Y≈const) — feeding raw coords gives vertical stripes. Rotate the
        # coords 90° about X so world-Z maps to the brick's Y → real blocks on the walls.
        bmap = nt.nodes.new("ShaderNodeMapping")
        bmap.inputs["Rotation"].default_value = (math.radians(90.0), 0.0, 0.0)
        nt.links.new(coords, bmap.inputs["Vector"])
        brick_coords = bmap.outputs["Vector"]
    # Rectangular mortared blocks (TexBrick) — matches the plate's masonry far better than
    # voronoi cobble. Two block tones + dark mortar; the Fac output is the mortar mask.
    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.offset = 0.5
    brick.offset_frequency = 2
    brick.squash = 0.86  # vary block widths so rows aren't a perfect repeat
    brick.squash_frequency = 2
    brick.inputs["Scale"].default_value = 2.6  # larger, fewer blocks (plate masonry)
    brick.inputs["Color1"].default_value = (*tuple(c * 0.85 for c in base_rgb), 1.0)
    brick.inputs["Color2"].default_value = (*tuple(min(1.0, c * 1.25) for c in base_rgb), 1.0)
    brick.inputs["Mortar"].default_value = (*tuple(c * 0.25 for c in base_rgb), 1.0)
    brick.inputs["Mortar Size"].default_value = 0.025
    if "Mortar Smooth" in brick.inputs:
        brick.inputs["Mortar Smooth"].default_value = 0.1
    brick.inputs["Brick Width"].default_value = 0.62
    brick.inputs["Row Height"].default_value = 0.42
    # Perturb texture coordinates to make the brick lines wavy/irregular/rustic
    perturb = nt.nodes.new("ShaderNodeTexNoise")
    perturb.name = "stone_perturb_noise"
    perturb.inputs["Scale"].default_value = 6.0
    perturb.inputs["Detail"].default_value = 4.0
    perturb.inputs["Distortion"].default_value = 0.5

    mix_co = nt.nodes.new("ShaderNodeMixRGB")
    mix_co.name = "stone_perturb_mix"
    mix_co.blend_type = "MIX"
    mix_co.inputs["Fac"].default_value = 0.15  # subtle wave/warp
    nt.links.new(brick_coords, mix_co.inputs[1])
    nt.links.new(perturb.outputs["Color"], mix_co.inputs[2])

    nt.links.new(mix_co.outputs["Color"], brick.inputs["Vector"])
    # Per-block tone variation so the wall isn't a flat repeat.
    var = nt.nodes.new("ShaderNodeTexNoise")
    var.inputs["Scale"].default_value = 3.0
    nt.links.new(coords, var.inputs["Vector"])
    tone = nt.nodes.new("ShaderNodeMixRGB")
    tone.blend_type = "MULTIPLY"
    tone.inputs["Fac"].default_value = 0.35  # stronger aged block-to-block tone variation
    nt.links.new(brick.outputs["Color"], tone.inputs["Color1"])
    nt.links.new(var.outputs["Color"], tone.inputs["Color2"])
    nt.links.new(tone.outputs["Color"], bsdf.inputs["Base Color"])
    # Deep recessed mortar grooves (brick Fac) + fine grit → blocky relief bump.
    grit = nt.nodes.new("ShaderNodeTexNoise")
    grit.inputs["Scale"].default_value = 28.0
    grit.inputs["Detail"].default_value = 4.0
    nt.links.new(coords, grit.inputs["Vector"])
    chip = nt.nodes.new("ShaderNodeTexVoronoi")
    chip.inputs["Scale"].default_value = 11.0
    if "Randomness" in chip.inputs:
        chip.inputs["Randomness"].default_value = 0.55
    nt.links.new(coords, chip.inputs["Vector"])
    bump_mortar = nt.nodes.new("ShaderNodeBump")
    bump_mortar.inputs["Strength"].default_value = 0.7
    bump_mortar.inputs["Distance"].default_value = 0.05
    bump_mortar.invert = True  # mortar sits recessed below the block faces
    nt.links.new(brick.outputs["Fac"], bump_mortar.inputs["Height"])
    bump_chip = nt.nodes.new("ShaderNodeBump")
    bump_chip.inputs["Strength"].default_value = 0.08
    bump_chip.inputs["Distance"].default_value = 0.015
    nt.links.new(chip.outputs["Distance"], bump_chip.inputs["Height"])
    nt.links.new(bump_mortar.outputs["Normal"], bump_chip.inputs["Normal"])
    bump_grit = nt.nodes.new("ShaderNodeBump")
    bump_grit.inputs["Strength"].default_value = 0.16
    bump_grit.inputs["Distance"].default_value = 0.01
    nt.links.new(grit.outputs["Fac"], bump_grit.inputs["Height"])
    nt.links.new(bump_chip.outputs["Normal"], bump_grit.inputs["Normal"])
    nt.links.new(bump_grit.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    # Real masonry relief: blocks stand proud, mortar recedes (invert brick Fac). With
    # adaptive subdivision (enabled at render) the key rakes across physical mortar lines
    # and block edges — the depth a bump map only fakes. Needs displacement_method=BOTH.
    disp = nt.nodes.new("ShaderNodeDisplacement")
    disp.inputs["Scale"].default_value = 0.05
    disp.inputs["Midlevel"].default_value = 0.5
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    nt.links.new(brick.outputs["Fac"], inv.inputs[1])
    grit_disp = nt.nodes.new("ShaderNodeMath")
    grit_disp.operation = "MULTIPLY_ADD"
    grit_disp.inputs[1].default_value = 0.15  # fine grit on the block faces
    nt.links.new(grit.outputs["Fac"], grit_disp.inputs[0])
    nt.links.new(inv.outputs["Value"], grit_disp.inputs[2])
    # Low-freq undulation so the whole wall swells/dips like settled old masonry — kills
    # the dead-flat perfect-grid CG read. Centred around 0 so it doesn't lift the midlevel.
    undu = nt.nodes.new("ShaderNodeTexNoise")
    undu.inputs["Scale"].default_value = 4.5
    undu.inputs["Detail"].default_value = 2.0
    nt.links.new(coords, undu.inputs["Vector"])
    undu_c = nt.nodes.new("ShaderNodeMapRange")
    undu_c.inputs["To Min"].default_value = -0.28
    undu_c.inputs["To Max"].default_value = 0.28
    nt.links.new(undu.outputs["Fac"], undu_c.inputs["Value"])
    height = nt.nodes.new("ShaderNodeMath")
    height.operation = "ADD"
    nt.links.new(grit_disp.outputs["Value"], height.inputs[0])
    nt.links.new(undu_c.outputs["Result"], height.inputs[1])
    nt.links.new(height.outputs["Value"], disp.inputs["Height"])
    nt.links.new(disp.outputs["Displacement"], out.inputs["Displacement"])
    if hasattr(mat, "displacement_method"):
        mat.displacement_method = "BOTH"
    ink = GRIMDARK_COLOURS["ink"]
    mat.line_color = ink[:4] if len(ink) > 3 else (*ink[:3], 1.0)
    mat.line_priority = 1
    return mat


def build_frosted_glass(
    name: str = "wood_glass",
    cool_rgb: tuple[float, float, float] = (0.60, 0.72, 1.0),
    *,
    emit_strength: float = 2.4,
) -> bpy.types.Material:
    """Frosted window: rough translucent BSDF + a cool emission backing so panes glow
    cold like moonlight (Epic 3 adds an area light behind them for the practical)."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.use_backface_culling = False
    nt = mat.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    transl = nt.nodes.new("ShaderNodeBsdfTranslucent")
    transl.inputs["Color"].default_value = (*cool_rgb, 1.0)
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (*cool_rgb, 1.0)
    emit.inputs["Strength"].default_value = emit_strength
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(transl.outputs["BSDF"], add.inputs[0])
    nt.links.new(emit.outputs["Emission"], add.inputs[1])
    # Frosted surface: noise bump so the glow breaks up rather than reading as a flat card.
    co = nt.nodes.new("ShaderNodeTexCoord")
    frost = nt.nodes.new("ShaderNodeTexNoise")
    frost.inputs["Scale"].default_value = 38.0
    frost.inputs["Detail"].default_value = 3.0
    nt.links.new(co.outputs["Object"], frost.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.01
    nt.links.new(frost.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], transl.inputs["Normal"])
    nt.links.new(add.outputs["Shader"], out.inputs["Surface"])
    return mat


_BUILT_WOOD: dict[str, bpy.types.Material] = {}

# Set meshes whose name matches these route to the dedicated stone / glass builders
# instead of the plank-wood material.
_STONE_KEYS = ("hearth", "stone", "oven", "chimney", "fireplace", "masonry")
_GLASS_KEYS = ("window", "pane", "glass")


def _wood_for(char: str | None, slot: str) -> bpy.types.Material:
    key = f"{char or 'base'}_{slot}"
    if key in _BUILT_WOOD:
        return _BUILT_WOOD[key]

    custom_tones = {
        "father_cap": (0.24, 0.16, 0.10),  # Dark brown leather
        "father_shirt": (0.74, 0.57, 0.37),  # Light wood base
        "father_pants": (0.30, 0.20, 0.12),  # Dark wood base
        "father_vest": (0.56, 0.42, 0.26),  # Medium wood base
        "hansel_shorts": (0.76, 0.58, 0.37),  # Bare skin-like wood base
        "hansel_shirt": (0.56, 0.42, 0.26),  # Clothing wood base
    }
    custom_paints = {
        "father_pants": (0.35, 0.25, 0.15),  # Brown pants
        "father_vest": (0.25, 0.25, 0.25),  # Dark grey vest
        "hansel_shirt": (0.22, 0.34, 0.18),  # Olive green tunic
    }

    tone = _WOOD_TONES.get(slot, _WOOD_TONES["cloth"])
    if slot in custom_tones:
        tone = custom_tones[slot]
    elif char and char in _WOOD_CHAR_TINT and slot in _WOOD_CHAR_TINT[char]:
        tone = _WOOD_CHAR_TINT[char][slot]

    # RINGS on a cylindrical limb reads as concentric zebra bands wrapping the leg.
    # Skin uses warped BANDS like the rest (organic carved figure); only eyes keep rings.
    grain_type = "rings" if slot == "eye" else "bands"
    grain_scale = 9.0 if slot == "skin" else 5.0
    # Hair/eyes: no engraving (cross-hatch on a round head reads as a wireframe net).
    # Skin: sparse single-direction hatch only, in the shadow planes.
    engrave = slot not in ("hair", "eye", "father_cap", "hansel_shorts")

    paint = None
    if slot in custom_paints:
        paint = custom_paints[slot]
    elif char and char in _WOOD_PAINT and slot in _WOOD_PAINT[char]:
        paint = _WOOD_PAINT[char][slot]
    mat = build_carved_wood(
        f"wood_{key}",
        tone,
        grain_type=grain_type,
        grain_scale=grain_scale,
        grain_contrast=0.32 if slot == "skin" else 0.5,
        grain_warp=0.18 if slot == "cloth" else (0.30 if slot == "skin" else 0.10),
        grain_rot=_GRAIN_ROT.get(slot, (0.0, 0.0, 0.0)),
        knots=(slot in ("cloth", "skin")),
        # Dense single-direction hatch over a large smooth torso reads as circuit-board
        # traces (a slop tell, e.g. shot13). Sparser, shallower grooves on skin keep the
        # carved-wood feel without the lattice; cloth keeps its woodcut hatch.
        hatch_scale=4.0 if slot == "skin" else 14.0,
        crosshatch=(slot == "cloth"),
        fine_hatch=(slot == "cloth"),
        engrave=engrave,
        bump_hatch_strength=0.06 if slot == "skin" else 0.13,
        paint_rgb=paint,
        face_bump=(slot == "skin"),
    )
    _BUILT_WOOD[key] = mat
    return mat


def assign_wood_to_meshes(objects: list[bpy.types.Object] | None = None) -> int:
    """Carved-wood materials on character meshes (per-char tone)."""
    _BUILT_WOOD.clear()
    targets = objects or [o for o in bpy.data.objects if o.type == "MESH"]
    count = 0
    for obj in targets:
        if _is_set_mesh(obj.name):
            continue
        char = _char_from_collections(obj)
        slot = _pick_slot(obj.name)

        # Specific sub-slot overrides to match target plate colors
        if char == "father":
            nm = obj.name.lower()
            if "helmet" in nm or "cap" in nm:
                slot = "father_cap"
            elif "shirt" in nm:
                slot = "father_shirt"
            elif "pants" in nm:
                slot = "father_pants"
            elif "vest" in nm:
                slot = "father_vest"
        elif char == "hansel":
            nm = obj.name.lower()
            if "short" in nm:
                slot = "hansel_shorts"
            elif "shirt" in nm:
                slot = "hansel_shirt"

        obj.data.materials.clear()
        obj.data.materials.append(_wood_for(char, slot))
        _shade_for_ink(obj)
        count += 1
    return count


def _set_material_kind(name: str) -> str:
    """Route a set mesh to 'glass' (windows), 'stone' (hearth/oven), or 'wood'."""
    nm = name.lower()
    if any(k in nm for k in _GLASS_KEYS):
        return "glass"
    if any(k in nm for k in _STONE_KEYS):
        return "stone"
    return "wood"


def assign_wood_to_set_meshes() -> int:
    """Carved-wood-plank / stone / frosted-glass materials on set meshes. Hearths and
    ovens get chunky stone; windows get frosted cool-glow glass; everything else is
    vertical plank grain + heavy hatch."""
    count = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        kind = _set_material_kind(obj.name)
        # Glass meshes (e.g. ext_window) are routed even when not a plank set mesh.
        if kind == "wood" and not _is_set_mesh(obj.name):
            continue
        if obj.data.library:
            try:
                obj.make_local()
            except Exception:
                pass
        # Apply scale so procedural textures and displacement are not stretched/flattened
        with bpy.context.temp_override(object=obj, active_object=obj):
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        if kind == "glass":
            key = "set_glass"
            if key not in _BUILT_WOOD:
                _BUILT_WOOD[key] = build_frosted_glass(f"wood_{key}")
            mat = _BUILT_WOOD[key]
        elif kind == "stone":
            if "floor" in obj.name.lower():
                key = "set_stone_floor"
                if key not in _BUILT_WOOD:
                    _BUILT_WOOD[key] = build_stone(f"wood_{key}", (0.24, 0.22, 0.20), is_floor=True)
            else:
                key = "set_stone"
                if key not in _BUILT_WOOD:
                    _BUILT_WOOD[key] = build_stone(
                        f"wood_{key}", (0.24, 0.22, 0.20), is_floor=False
                    )
            mat = _BUILT_WOOD[key]
        else:
            tone = _set_tone_for(obj.name)
            # Scale set tone up into a visible wood range (set tones are very dark).
            tone = tuple(min(1.0, c * 2.2 + 0.10) for c in tone)
            nm = obj.name.lower()
            is_ext = obj.name.startswith("ext_") or (
                bool(obj.data.library) and "exterior" in str(obj.data.library.filepath).lower()
            )
            # Trunks/posts get vertical grain; grounds/floors horizontal; exterior wood
            # gets cranked engraving contrast for a white-on-black woodcut read.
            if any(k in nm for k in ("trunk", "tree", "log", "post", "pole")):
                set_rot = (0.0, math.pi * 0.5, 0.0)
            else:
                set_rot = (0.0, 0.0, 0.0)
            # Big flat planes (walls/roof) at the fine hatch scale read as a corrugated
            # zebra/woodcut weave — the #1 background slop tell. Give them broad, calm
            # plank grain and NO engraving so they sit back as boards, not ribbing.
            is_wall = any(k in nm for k in ("wall", "roof"))
            set_contrast = 0.45 if is_ext else 0.22
            key = f"set_{'ext_' if is_ext else ''}{obj.name}"
            if key not in _BUILT_WOOD:
                _BUILT_WOOD[key] = build_carved_wood(
                    f"wood_{key}",
                    tone,
                    grain_type="bands",
                    grain_scale=2.5 if is_wall else 9.0,
                    grain_contrast=0.12 if is_wall else set_contrast,
                    grain_rot=set_rot,
                    hatch_scale=5.0 if is_ext else 4.0,
                    crosshatch=is_ext and not is_wall,
                    engrave=not is_wall,
                )
            mat = _BUILT_WOOD[key]
        obj.data.materials.clear()
        if not obj.data.materials:
            obj.data.materials.append(None)
        obj.data.materials[0] = mat
        _shade_for_ink(obj)
        count += 1
    return count


def _stabilize_freestyle_lines(scene: bpy.types.Scene, thickness: float = 3.0) -> None:
    """Hand-inked woodcut lines: calligraphic thick/thin + pressure noise + gentle
    wobble. A flat uniform stroke is the #1 tell of a CAD render; real block-print
    contours swell and taper. Thick on silhouettes, tapering into creases."""
    if not scene.render.use_freestyle:
        return
    ls = scene.view_layers[0].freestyle_settings.linesets[0]
    style = ls.linestyle
    style.thickness = thickness
    style.thickness_position = "CENTER"

    # Crisp, confident outline. A subtle calligraphic swell gives the line life, but
    # NO pressure noise and NO geometric wobble — those read as a shaky melted scrawl.
    names = [m.name for m in style.thickness_modifiers]
    if "ink_calligraphy" not in names:
        cal = style.thickness_modifiers.new(name="ink_calligraphy", type="CALLIGRAPHY")
        cal.orientation = 0.7854  # 45° nib
        cal.thickness_min = thickness * 0.70
        cal.thickness_max = thickness * 1.25
    for m in list(style.thickness_modifiers):
        if m.name == "ink_pressure":
            style.thickness_modifiers.remove(m)
    for mod in style.geometry_modifiers:
        if mod.type == "SINUS_DISPLACEMENT":
            mod.amplitude = 0.0
            mod.wavelength = 1.0


def _mood_color_cast(mood: str) -> tuple[float, float, float]:
    """Final-grade color cast multiplied over the rendered frame so each mood
    has a distinct cinematic tint beyond the material palette."""
    return {
        "moon": (0.85, 0.92, 1.05),
        "ember": (1.08, 0.92, 0.78),
        "noon": (1.02, 1.00, 0.95),
        "dusk": (1.10, 0.85, 0.72),
    }.get(mood, (1.0, 1.0, 1.0))


def _apply_compositor_grade(scene: bpy.types.Scene) -> None:
    """Print-pass grade: tight vignette, mood cast, sepia desaturation, woodgrain
    paper-fibre overlay, and grain. Turns the flat EEVEE frame into something that
    reads as ink pressed onto aged paper rather than a clean digital render."""
    scene.use_nodes = True
    nt = scene.node_tree
    for n in list(nt.nodes):
        if n.name.startswith("grimdark_grade_"):
            nt.nodes.remove(n)
    render_layers = next((n for n in nt.nodes if n.type == "R_LAYERS"), None)
    output = next((n for n in nt.nodes if n.type == "COMPOSITE"), None)
    if render_layers is None or output is None:
        return
    for link in list(nt.links):
        if link.to_node == output and link.to_socket == output.inputs["Image"]:
            nt.links.remove(link)

    # 1. Vignette — tighter ellipse (1.1) so corners fall off harder, framing the figure.
    ellipse = nt.nodes.new("CompositorNodeEllipseMask")
    ellipse.name = "grimdark_grade_ellipse"
    ellipse.width = 1.1
    ellipse.height = 1.1
    blur = nt.nodes.new("CompositorNodeBlur")
    blur.name = "grimdark_grade_blur"
    blur.size_x = 380
    blur.size_y = 380
    nt.links.new(ellipse.outputs["Mask"], blur.inputs["Image"])
    mul = nt.nodes.new("CompositorNodeMixRGB")
    mul.name = "grimdark_grade_mul"
    mul.blend_type = "MULTIPLY"
    mul.inputs["Fac"].default_value = 0.28
    nt.links.new(render_layers.outputs["Image"], mul.inputs[1])
    nt.links.new(blur.outputs["Image"], mul.inputs[2])

    # 2. Per-mood color cast.
    cast_r, cast_g, cast_b = _mood_color_cast(_ACTIVE_MOOD)
    rgb_cast = nt.nodes.new("CompositorNodeRGB")
    rgb_cast.name = "grimdark_grade_cast"
    rgb_cast.outputs["RGBA"].default_value = (cast_r, cast_g, cast_b, 1.0)
    cast_mul = nt.nodes.new("CompositorNodeMixRGB")
    cast_mul.name = "grimdark_grade_cast_mul"
    cast_mul.blend_type = "MULTIPLY"
    cast_mul.inputs["Fac"].default_value = 0.55
    nt.links.new(mul.outputs["Image"], cast_mul.inputs[1])
    nt.links.new(rgb_cast.outputs["RGBA"], cast_mul.inputs[2])

    # 3. Light desaturation only — keep the palette controlled but NOT a brown
    #    sepia wash (that mud was a big part of the melted-wax read).
    desat = nt.nodes.new("CompositorNodeHueSat")
    desat.name = "grimdark_grade_desat"
    # Mild RE-saturation (AgX keeps more chroma than Filmic, and the albedo now carries
    # the warmth) — kept modest so it never tips into the sepia/candy over-grade.
    desat.inputs["Saturation"].default_value = 1.05
    nt.links.new(cast_mul.outputs["Image"], desat.inputs["Image"])

    # 4. Gentle bloom on the bright practicals (candle/window) — high threshold so only
    #    the light sources glow, framing them as warm sources without hazing the wood.
    glare = nt.nodes.new("CompositorNodeGlare")
    glare.name = "grimdark_grade_bloom"
    glare.glare_type = "FOG_GLOW"
    glare.quality = "MEDIUM"
    glare.threshold = 1.0
    glare.size = 7
    if hasattr(glare, "mix"):
        glare.mix = -0.4  # subtle add, not a blown-out haze
    nt.links.new(desat.outputs["Image"], glare.inputs["Image"])

    # 5. Whisper of film grain so it isn't clinically flat. No paper/woodgrain
    #    overlay — that screen-space texture muddied every surface. No sepia.
    grain = nt.nodes.new("CompositorNodeTexture")
    grain.name = "grimdark_grade_grain_src"
    tex = bpy.data.textures.get("grimdark_grain")
    if tex is None:
        tex = bpy.data.textures.new("grimdark_grain", type="NOISE")
    grain.texture = tex
    mix_grain = nt.nodes.new("CompositorNodeMixRGB")
    mix_grain.name = "grimdark_grade_grain_mix"
    mix_grain.blend_type = "OVERLAY"
    mix_grain.inputs["Fac"].default_value = 0.05
    nt.links.new(glare.outputs["Image"], mix_grain.inputs[1])
    nt.links.new(grain.outputs["Color"], mix_grain.inputs[2])
    nt.links.new(mix_grain.outputs["Image"], output.inputs["Image"])


def apply_grimdark_view(
    scene: bpy.types.Scene | None = None,
    *,
    fast: bool = False,
    line_thickness: float = 3.0,
) -> None:
    """Production render preset; optional fast probe resolution."""
    scene = scene or bpy.context.scene
    apply_grimdark_render_settings(scene=scene)
    # Look names are view-transform-specific: "High Contrast" is Filmic-only and silently
    # no-ops/errors under AgX. Try AgX-punchy contrast first, then generic, then none.
    for _look in ("AgX - Punchy", "AgX - High Contrast", "High Contrast", "None"):
        try:
            scene.view_settings.look = _look
            break
        except TypeError:
            continue
    scene.view_settings.exposure = -0.15
    scene.view_settings.gamma = 1.1
    _apply_compositor_grade(scene)
    _stabilize_freestyle_lines(scene, line_thickness)
    if scene.world and scene.world.node_tree:
        for node in scene.world.node_tree.nodes:
            if node.type == "BACKGROUND":
                node.inputs["Strength"].default_value = 0.35
    if fast:
        scene.render.resolution_x = 1280
        scene.render.resolution_y = 540
        scene.eevee.taa_render_samples = 32
        if hasattr(scene.eevee, "use_raytracing"):
            scene.eevee.use_raytracing = False
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = True
            scene.eevee.gtao_distance = 0.35
            scene.eevee.gtao_factor = 2.0
            scene.eevee.gtao_quality = 0.5
