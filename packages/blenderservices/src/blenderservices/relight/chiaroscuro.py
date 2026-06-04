"""Relight a shot for carved-wood chiaroscuro: one warm key, crushed fill, near-black
world, and a volumetric mist so the key reads as a god-ray cone (matches the
style_targets plates, e.g. shot10 — single overhead warm pool, deep-black falloff).

Flat even lighting is the biggest remaining "videogame" tell. This pass picks the warmest /
strongest existing light as the key, boosts it, crushes the rest to a whisper of fill —
except the *coolest* light, which is kept as a dim blue rim so the dark side reads as cool
moonlight rather than dead black (true dual-tone chiaroscuro). Window meshes get a cool
area light behind them so frosted glass glows. A near-black world volume scatters the key.

Interior shots (default):
    blender --background <shot>.blend --python -m blenderservices.relight.chiaroscuro -- \\
        --preset interior

Exterior night plates (dark woodcut forest, shot01/18):
    blender --background <shot>.blend --python -m blenderservices.relight.chiaroscuro -- \\
        --preset moon_engrave
"""

from __future__ import annotations

import argparse
import math

import bpy
from mathutils import Euler, Vector

from blenderservices.bpy_common import blender_argv

WARM = (1.0, 0.72, 0.42)
COOL = (0.55, 0.66, 1.0)
KEY_ENERGY_SUN = 5.0
KEY_ENERGY_POINT = 220.0
KEY_ENERGY_SPOT = 400.0
FILL_MULT = 0.04  # near-kill fill so shadows fall to black (true chiaroscuro)
COOL_RIM = 0.40  # keep the coolest light as a dim blue rim at this fraction of its energy
MIST_DENSITY = 0.009

MOON_KEY_SUN = 3.2  # cool moon key for exterior night plates

# Per-shot key/mist overrides. A single global preset is generic; close interiors want
# tight low mist, big rooms a heavier cone. key_mult scales the chosen key energy.
_SHOT_TUNING: dict[str, dict[str, float]] = {
    "shot06": {"mist": 0.006, "key_mult": 1.0},  # table candle, close interior
    "shot10": {"mist": 0.012, "key_mult": 1.25},  # overhead pool, big room
    "shot14": {"mist": 0.008, "key_mult": 1.0},  # two kids, mid interior
    # Hero: a motivated fireplace POINT already rakes from frame-right — use it as the
    # hard key instead of a synthetic centred candle (no_candle), for true side-chiaroscuro.
    "shot07_hero": {"mist": 0.003, "key_mult": 0.6, "no_candle": 1.0},
}

_WINDOW_KEYS = ("window", "pane", "glass")
_SET_KEYS = (
    "wall",
    "floor",
    "ceiling",
    "roof",
    "ground",
    "hut_",
    "set_",
    "hearth",
    "bed_",
    "stool",
    "bench",
    "table",
    "ext_",
)
CANDLE_ENERGY = 22.0  # close point light — inverse-square means this models a figure
# without blowing out the near wall (the distant-key 220 W washes everything flat).


def _shot_id() -> str:
    from pathlib import Path

    return Path(bpy.data.filepath).stem.lower()


def _character_centroid():
    """Centre of the character meshes (sets excluded) with z at chest height. A
    candle placed here lights the actors, not the room geometry / floor."""
    pts: list[Vector] = []
    for o in bpy.context.scene.objects:
        if o.type != "MESH":
            continue
        nm = o.name.lower()
        if any(k in nm for k in _SET_KEYS) or any(k in nm for k in _WINDOW_KEYS):
            continue
        for corner in o.bound_box:
            pts.append(o.matrix_world @ Vector(corner))
    if not pts:
        return Vector((0.0, 0.0, 1.2))
    cx = sum(p.x for p in pts) / len(pts)
    cy = sum(p.y for p in pts) / len(pts)
    zmin = min(p.z for p in pts)
    zmax = max(p.z for p in pts)
    chest = zmin + (zmax - zmin) * 0.62
    # Pull the candle toward camera (-Y) and up a touch so the key rakes across faces.
    return Vector((cx, cy - 0.55, chest + 0.05))


def _ensure_candle_practical():
    """Warm point light + tiny ember bead at the character centroid (chest height) so
    every interior has a motivated subject key. Idempotent. Returns the light object."""
    for name in ("practical_candle", "practical_candle_flame"):
        old = bpy.data.objects.get(name)
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)
    loc = _character_centroid()
    bpy.ops.object.light_add(type="POINT", location=tuple(loc))
    lo = bpy.context.object
    lo.name = "practical_candle"
    lo.data.energy = CANDLE_ENERGY
    lo.data.color = (1.0, 0.66, 0.32)
    if hasattr(lo.data, "shadow_soft_size"):
        lo.data.shadow_soft_size = 0.12
    # Tiny emissive bead so the source itself reads as a flame in-frame.
    mesh = bpy.data.meshes.new("practical_candle_flame")
    flame = bpy.data.objects.new("practical_candle_flame", mesh)
    bpy.context.scene.collection.objects.link(flame)
    import bmesh

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=0.03)
    bm.to_mesh(mesh)
    bm.free()
    flame.location = loc
    fmat = bpy.data.materials.new("practical_candle_mat")
    fmat.use_nodes = True
    fnt = fmat.node_tree
    for n in list(fnt.nodes):
        fnt.nodes.remove(n)
    fout = fnt.nodes.new("ShaderNodeOutputMaterial")
    femit = fnt.nodes.new("ShaderNodeEmission")
    femit.inputs["Color"].default_value = (1.0, 0.7, 0.34, 1.0)
    femit.inputs["Strength"].default_value = 12.0
    fnt.links.new(femit.outputs["Emission"], fout.inputs["Surface"])
    flame.data.materials.append(fmat)
    return lo


def _warmth(light) -> float:
    c = light.data.color
    return c[0] - c[2]  # red-minus-blue: warmer = higher


def _pick_key(lights):
    """Prefer an explicit practical (candle/hearth/ember/lamp), else warmest, else strongest."""
    keys = [o for o in lights if "key" in o.name.lower() and "window" not in o.name.lower()]
    if keys:
        return max(keys, key=lambda o: o.data.energy)
    practicals = [
        o
        for o in lights
        if any(k in o.name.lower() for k in ("practical", "candle", "hearth", "ember", "lamp"))
    ]
    pool = practicals or lights
    return max(pool, key=lambda o: (_warmth(o), o.data.energy), default=None)


def _pick_cool_rim(lights, key):
    """Coolest non-key light — kept as a dim blue rim rather than crushed to black."""
    pool = [o for o in lights if o is not key]
    if not pool:
        return None
    coolest = min(pool, key=_warmth)
    # Only treat it as a rim if it is actually cool-ish; otherwise it's just fill.
    return coolest if _warmth(coolest) < 0.05 else None


def _black_world(
    mist: float,
    *,
    ambient: float = 0.03,
    ambient_color: tuple[float, float, float] = (0.05, 0.04, 0.03),
) -> None:
    """Near-black world with a whisper of ambient so the subject lifts out of pure
    void (a strength-0 world crushed every off-key surface to dead black). Keeps the
    deep chiaroscuro falloff but readable. Plus a low-density volume for god-ray scatter."""
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("chiaro_world")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (*ambient_color, 1.0)
    bg.inputs["Strength"].default_value = ambient
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    # Atmospheric mist so the key scatters into a visible cone.
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (0.55, 0.45, 0.35, 1.0)
    vol.inputs["Density"].default_value = mist
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])


def _add_window_lights() -> int:
    """Place a cool area light just behind every window mesh so frosted glass glows
    cold like moonlight. Idempotent: replaces prior 'practical_window_*' lights."""
    scene = bpy.context.scene
    for o in list(scene.objects):
        if o.type == "LIGHT" and o.name.startswith("practical_window_"):
            bpy.data.objects.remove(o, do_unlink=True)
    windows = [
        o
        for o in scene.objects
        if o.type == "MESH" and any(k in o.name.lower() for k in _WINDOW_KEYS)
    ]
    added = 0
    for win in windows:
        loc = win.matrix_world.translation
        outward = loc.copy()
        _EPS = 1e-6
        if outward.length < _EPS:
            outward = Vector((0.0, 1.0, 0.0))
        else:
            outward.normalize()
        light_loc = loc + outward * 0.4
        bpy.ops.object.light_add(type="AREA", location=tuple(light_loc))
        lo = bpy.context.object
        lo.name = f"practical_window_{win.name}"
        lo.data.energy = 30.0
        lo.data.color = (0.6, 0.7, 1.0)
        lo.data.size = 0.8
        direction = loc - light_loc
        if direction.length < _EPS:
            direction = Vector((0.0, 0.0, -1.0))
        lo.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        added += 1
    return added


WARM_FILL = 0.06  # warm practicals (hearth/ember) survive as dim motivated fill


def _relight(lights, key, *, key_color, key_mult: float) -> None:
    cool_rim = _pick_cool_rim(lights, key)
    for o in lights:
        ld = o.data
        if o is key:
            ld.color = key_color
            if ld.type == "SUN":
                ld.energy = KEY_ENERGY_SUN * key_mult
                ld.angle = 0.09
            elif ld.type == "SPOT":
                ld.energy = KEY_ENERGY_SPOT * key_mult
                ld.spot_size = 1.1
                ld.shadow_soft_size = 0.25
            elif o.name == "practical_candle":
                # Close candle key: keep its own modest energy (KEY_ENERGY_POINT is a
                # distant-key value that blows out a near subject + wall).
                ld.energy = CANDLE_ENERGY * key_mult
                if hasattr(ld, "shadow_soft_size"):
                    ld.shadow_soft_size = 0.12
            else:  # POINT / AREA
                ld.energy = max(ld.energy, KEY_ENERGY_POINT) * key_mult
                if hasattr(ld, "shadow_soft_size"):
                    ld.shadow_soft_size = 0.08
            print(f"  key: {o.name} ({ld.type}) e={ld.energy:.1f}")
        elif o is cool_rim:
            ld.color = COOL
            ld.energy *= COOL_RIM
            print(f"  cool rim kept: {o.name} -> {ld.energy:.1f}")
        elif _warmth(o) > 0.15:
            # Keep warm practicals (hearth/ember glow) as a dim motivated fill rather
            # than crushing them — they add the second warm pool the plates show.
            ld.energy *= WARM_FILL
            print(f"  warm fill kept: {o.name} -> {ld.energy:.1f}")
        else:
            ld.energy *= FILL_MULT
            print(f"  fill crushed: {o.name} -> {ld.energy:.1f}")


def apply_chiaroscuro(*, save: bool = True) -> None:
    scene = bpy.context.scene
    lights = [o for o in scene.objects if o.type == "LIGHT"]
    if not lights:
        bpy.ops.object.light_add(type="SPOT", location=(0.4, -0.6, 2.6))
        lights = [bpy.context.object]

    sid = _shot_id()
    tuning = _SHOT_TUNING.get(sid, {})
    mist = tuning.get("mist", MIST_DENSITY)
    key_mult = tuning.get("key_mult", 1.0)
    no_candle = bool(tuning.get("no_candle", 0.0))

    # A motivated candle at the character centroid is the guaranteed subject key — the
    # shots' authored lights often miss the actors once fill is crushed to chiaroscuro.
    # Shots that already own a strong, well-placed practical (no_candle) skip it and let
    # _pick_key choose that warm source for a harder, directional key.
    candle = None if no_candle else _ensure_candle_practical()
    lights = [o for o in scene.objects if o.type == "LIGHT"]
    key = candle if candle is not None else _pick_key(lights)
    _relight(lights, key, key_color=WARM, key_mult=key_mult)
    n_win = _add_window_lights()
    if n_win:
        print(f"  window practicals: {n_win}")
    _black_world(mist)
    if save:
        bpy.ops.wm.save_mainfile()
        print("✓ chiaroscuro relight applied and saved.")
    else:
        print("✓ chiaroscuro relight applied (not saved).")


def apply_moon_engrave(*, save: bool = True) -> None:
    """Exterior night plate: cool moon key, near-black world, dim warm fill kept off.
    The single warm accent (hut window glow) is added separately by stage dressing /
    the exterior window light. Ground/trunk hatch direction is handled in the wood
    set-material builder."""
    scene = bpy.context.scene
    suns = [o for o in scene.objects if o.type == "LIGHT" and o.data.type == "SUN"]
    # Reuse the strongest existing sun as the moon — it is already aimed at the scene.
    # Adding a fresh sun with an arbitrary rake just shadows the camera-facing walls.
    key = max(suns, key=lambda o: o.data.energy) if suns else _add_moon_sun()
    key.data.color = COOL
    key.data.energy = max(key.data.energy, MOON_KEY_SUN)
    key.data.angle = 0.06
    for o in scene.objects:
        if o.type == "LIGHT" and o is not key and not o.name.startswith("practical_"):
            o.data.energy *= FILL_MULT
    print(f"  moon key: {key.name} (SUN) e={key.data.energy:.1f}")
    _add_window_lights()
    _add_hut_glow()
    # Cool ambient so the hut/forest reads as moonlit engraving, not a black void.
    _black_world(MIST_DENSITY * 1.3, ambient=0.12, ambient_color=(0.05, 0.06, 0.11))
    if save:
        bpy.ops.wm.save_mainfile()
        print("✓ moon-engrave relight applied and saved.")
    else:
        print("✓ moon-engrave relight applied (not saved).")


def _add_moon_sun():
    """Add (or refresh) a cool moon-key SUN raking down across the scene. A sun's
    direction is its rotation, so we tilt it rather than relying on a position."""
    old = bpy.data.objects.get("moon_key")
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)
    bpy.ops.object.light_add(type="SUN")
    lo = bpy.context.object
    lo.name = "moon_key"
    lo.data.energy = MOON_KEY_SUN
    lo.data.color = COOL
    lo.data.angle = 0.06
    # Rake from high and to one side (down + 40° azimuth) for engraved tree-trunk shadows.
    lo.rotation_euler = Euler((math.radians(52.0), 0.0, math.radians(38.0)), "XYZ")
    return lo


def _add_hut_glow() -> None:
    """Single warm ember accent at a hut window — the one bright note in the dark
    exterior plate (shot01). Idempotent."""
    old = bpy.data.objects.get("practical_hut_glow")
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)
    glow_keys = (*_WINDOW_KEYS, "door")
    win = next(
        (
            o
            for o in bpy.context.scene.objects
            if o.type == "MESH" and any(k in o.name.lower() for k in glow_keys)
        ),
        None,
    )
    loc = win.matrix_world.translation if win is not None else None
    if loc is None:
        return
    bpy.ops.object.light_add(type="POINT", location=tuple(loc))
    lo = bpy.context.object
    lo.name = "practical_hut_glow"
    lo.data.energy = 60.0
    lo.data.color = (1.0, 0.6, 0.26)
    if hasattr(lo.data, "shadow_soft_size"):
        lo.data.shadow_soft_size = 0.15


def relight(preset: str = "interior", *, save: bool = True) -> None:
    """Dispatch entry usable from other scripts (e.g. the style probe's --relight)."""
    if preset == "moon_engrave":
        apply_moon_engrave(save=save)
    else:
        apply_chiaroscuro(save=save)


def main() -> None:
    argv = blender_argv()
    p = argparse.ArgumentParser(description="Carved-wood chiaroscuro relight")
    p.add_argument(
        "--preset",
        choices=("interior", "moon_engrave"),
        default="interior",
        help="interior = warm key + cool rim; moon_engrave = cool exterior night plate",
    )
    p.add_argument("--no-save", action="store_true", help="Relight in-memory only")
    args = p.parse_args(argv)
    relight(args.preset, save=not args.no_save)


if __name__ == "__main__":
    main()
