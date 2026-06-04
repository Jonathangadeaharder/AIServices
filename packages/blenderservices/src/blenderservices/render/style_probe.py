"""Render one still for fast GrimDark look-dev iteration.

Use on a character blend, shot blend, or the material lookdev file.

Examples:
    blender --background assets_3d/characters/father/father.blend \\
        --python -m blenderservices.render.style_probe -- \\
        --output /tmp/father_probe.png --frame 1 --wood --fast

    blender --background shot_production/ep01/reels/reel01/shots/shot04.blend \\
        --python -m blenderservices.render.style_probe -- \\
        --output /tmp/shot04_mid.png --frame 384 --fast
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import bpy
from mathutils import Vector

from blenderservices.bpy_common import blender_argv
from blenderservices.materials.grimdark import (
    apply_grimdark_view,
    assign_wood_to_meshes,
    assign_wood_to_set_meshes,
    enable_set_microdisplacement,
    set_active_mood,
)


def _apply_cycles(scene, *, fast: bool) -> None:
    """Cycles hero path: real wood shading, volumetrics, DOF. Slower than EEVEE."""
    scene.render.engine = "CYCLES"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.get_devices()
        for dev_type in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
            try:
                prefs.compute_device_type = dev_type
                break
            except TypeError:
                continue
        for d in prefs.devices:
            d.use = True
        scene.cycles.device = "GPU"
    except Exception as exc:  # noqa: BLE001
        print(f"   cycles GPU setup skipped: {exc}")
    scene.cycles.samples = 96 if fast else 256
    scene.cycles.use_denoising = True
    scene.cycles.volume_max_steps = 128
    # Modest lift only: the chiaroscuro key is the sole light, so a big push floods the
    # figures into a pale wash. Low-key keeps saturated wood dark/warm and lets shadows
    # fall toward black (target is low-key, not flat-bright). Trim knob for the grade.
    scene.view_settings.exposure = 0.40


def _detect_mood() -> str:
    """Pick mood tint from the brightest active light's color."""
    best = None
    best_energy = 0.0
    for obj in bpy.context.scene.objects:
        if obj.type != "LIGHT" or obj.hide_render:
            continue
        e = obj.data.energy
        if e <= 0.0:
            continue
        if e > best_energy:
            best_energy = e
            best = obj
    if best is None:
        return "neutral"
    name = best.name.lower()
    if "ember" in name or "hearth" in name or "candle" in name:
        return "ember"
    if "moon" in name:
        return "moon"
    r, g, b = best.data.color[:3]
    if r > g and r > b and r - b > 0.25:
        return "ember"
    if b > r and b - r > 0.15:
        return "moon"
    return "noon"


def _keep_shot_camera() -> bool:
    """Respect production cameras; only auto-frame look-dev / character blends."""
    cam = bpy.context.scene.camera
    if cam is None:
        return False
    name = cam.name.lower()
    if name.startswith("cam_"):
        return True
    return name not in ("camera", "probe_camera")


def ensure_preview_camera() -> None:
    if _keep_shot_camera():
        return
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("no mesh objects to frame")

    mins = Vector((1e9, 1e9, 1e9))
    maxs = Vector((-1e9, -1e9, -1e9))
    for obj in meshes:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            mins = Vector(min(mins[i], world[i]) for i in range(3))
            maxs = Vector(max(maxs[i], world[i]) for i in range(3))
    center = (mins + maxs) * 0.5
    height = max(maxs.z - mins.z, 0.5)
    width = max(maxs.x - mins.x, 0.4)

    cam = bpy.context.scene.camera
    if cam is None:
        bpy.ops.object.camera_add()
        cam = bpy.context.object
        cam.name = "probe_camera"
    cam.data.lens = 55
    cam.data.sensor_fit = "AUTO"
    fov = 2.0 * math.atan(18.0 / (2.0 * cam.data.lens))
    dist = max(height * 1.25, width * 1.1) / math.tan(fov * 0.5)
    cam.location = center + Vector((width * 0.08, -dist, height * 0.02))
    direction = center - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam

    lights = [o for o in bpy.context.scene.objects if o.type == "LIGHT"]
    if not lights:
        bpy.ops.object.light_add(
            type="SUN", location=center + Vector((width * 0.5, -dist * 0.55, height * 1.6))
        )
        key = bpy.context.object
        key.name = "probe_key"
        key.data.energy = 6.0
        key.data.angle = 0.02
        direction = center - key.location
        key.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        bpy.ops.object.light_add(
            type="SUN", location=center + Vector((-width, dist * 0.25, height * 0.5))
        )
        rim = bpy.context.object
        rim.name = "probe_rim"
        rim.data.energy = 1.2
        rim.data.angle = 0.08


def render_still(output: Path, frame: int) -> None:
    scene = bpy.context.scene
    scene.frame_set(frame)
    if scene.camera is None:
        cams = [o for o in scene.objects if o.type == "CAMERA"]
        if not cams:
            raise RuntimeError("no camera in scene")
        scene.camera = cams[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output.resolve())
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)
    print(f"✓ {output}")


def main() -> None:
    argv = blender_argv()
    parser = argparse.ArgumentParser(description="Single-frame GrimDark style probe")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument(
        "--wood",
        action="store_true",
        help="Assign carved-wood materials (procedural grain + engraved cross-hatch)",
    )
    parser.add_argument(
        "--engine",
        choices=("eevee", "cycles"),
        default="eevee",
        help="Render engine (cycles = hero carved-wood look with volumetrics/DOF)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="1280x540, 16 TAA samples (still keeps Freestyle)",
    )
    parser.add_argument(
        "--freestyle-thickness",
        type=float,
        default=None,
        help="Override Freestyle line thickness (default: preset 2.4)",
    )
    parser.add_argument(
        "--mood",
        choices=("moon", "ember", "noon", "neutral", "auto"),
        default="auto",
        help="Compositor mood tint (auto = detect from scene lights)",
    )
    parser.add_argument(
        "--no-freestyle",
        action="store_true",
        help="Disable Freestyle line render (huge speed-up on dense exterior scenes)",
    )
    parser.add_argument(
        "--relight",
        choices=("none", "interior", "moon_engrave"),
        default="none",
        help="Apply carved-wood chiaroscuro in-memory before rendering (non-destructive)",
    )
    args = parser.parse_args(argv)

    mood = args.mood if args.mood != "auto" else _detect_mood()
    set_active_mood(mood)

    if args.wood:
        n = assign_wood_to_meshes()
        s = assign_wood_to_set_meshes()
        print(f"   carved-wood · {n} char mesh(es), {s} set mesh(es) · mood={mood}")

    thickness = args.freestyle_thickness if args.freestyle_thickness is not None else 3.0
    apply_grimdark_view(fast=args.fast, line_thickness=thickness)
    if args.engine == "cycles":
        _apply_cycles(bpy.context.scene, fast=args.fast)
        if args.wood:
            k = enable_set_microdisplacement(dicing=3.0 if args.fast else 1.5)
            print(f"   microdisplacement · {k} static set/prop mesh(es)")
    if args.relight != "none":
        # After apply_grimdark_view/_apply_cycles so the black chiaroscuro world isn't
        # overwritten by the preset's ambient background. Never saves the blend.
        from blenderservices.relight.chiaroscuro import relight

        relight(args.relight, save=False)
    if args.no_freestyle:
        bpy.context.scene.render.use_freestyle = False
    if not _keep_shot_camera():
        ensure_preview_camera()

    render_still(args.output, args.frame)


if __name__ == "__main__":
    main()
