"""Palette materials — flat colour swatches with Freestyle ink marking."""

from __future__ import annotations

import bpy

# The seven canonical colours (matches Volume II spec exactly).
GRIMDARK_COLOURS = {
    "skin_base": (0.922, 0.890, 0.816, 1.0),  # #EBE3D0
    "skin_mid": (0.722, 0.659, 0.502, 1.0),  # #B8A880
    "skin_deep": (0.353, 0.302, 0.212, 1.0),  # #5A4D36
    "ink": (0.059, 0.051, 0.039, 1.0),  # #0F0D0A
    "ink_shadow": (0.110, 0.094, 0.075, 1.0),  # #1C1813
    "blood": (0.545, 0.102, 0.102, 1.0),  # #8B1A1A
    "ember": (0.761, 0.220, 0.039, 1.0),  # #C2380A — restricted use
}


def build_flat_material(name: str, base_color, line_color=None):
    """Build a simple Blender-safe palette swatch material.

    The production look is assigned by ``blenderservices.materials.grimdark`` at
    import/render time. This fallback only gives generated blockouts stable,
    linkable material names without carrying an obsolete EEVEE ramp graph.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.use_backface_culling = True
    nt = mat.node_tree

    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = tuple(base_color)
    bsdf.inputs["Roughness"].default_value = 0.82
    for spec_name in ("Specular IOR Level", "Specular"):
        if spec_name in bsdf.inputs:
            bsdf.inputs[spec_name].default_value = 0.05
            break
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # stash line colour as a custom property for the freestyle layer
    mat["grimdark_line_color"] = list(line_color or GRIMDARK_COLOURS["ink"])

    # enable freestyle marking
    mat.line_color = tuple(line_color or GRIMDARK_COLOURS["ink"])
    mat.line_priority = 1

    return mat


def assign_material(obj, material, slot: int = 0) -> None:
    """Push a material into a specific slot of an object (creating slots as needed)."""
    while len(obj.data.materials) <= slot:
        obj.data.materials.append(None)
    obj.data.materials[slot] = material
