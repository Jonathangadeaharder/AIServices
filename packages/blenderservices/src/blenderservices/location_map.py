"""Map legacy story-beat location and character IDs to assets_3d catalog IDs."""

from __future__ import annotations

import re
import unicodedata

# Direct renames from project.forge.yaml / legacy scripts → 3D catalog.
LEGACY_SET_MAP: dict[str, str] = {
    "oak_root_cave": "root_shelter",
    "crone_farm_exterior": "gingerbread_house_exterior",
    "crone_farm_interior": "gingerbread_house_interior",
    "crone_cellar": "cellar",
    "crone_oven_alcove": "oven_room",
    "iron_stove_clearing": "oven_room",
    "gingerbread_house": "gingerbread_house_exterior",
    "wood": "forest_path",
    "hut": "hut_interior",
}

LEGACY_CHARACTER_MAP: dict[str, str] = {
    "mother": "stepmother",
}

# Title keywords (lowercase, ASCII-normalized) → set override when YAML location is vague.
_TITLE_SET_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("dach", "regen", "waldrand", "felder"), "hut_exterior"),
    (
        (
            "tisch",
            "suppe",
            "asche",
            "stiefmutter",
            "bett",
            "vater",
            "kiesel",
            "mondlicht",
            "schussel",
            "kessel",
            "decke",
            "stroh",
            "axt",
            "abendbruhe",
            "flustern",
            "mehl",
            "loswerden",
            "nickt",
        ),
        "hut_interior",
    ),
    (("wurzel", "wurzeln"), "root_shelter"),
    (("haus", "pfefferkuchen", "schwelle", "tur"), "gingerbread_house_exterior"),
    (("stimme", "innen"), "gingerbread_house_interior"),
)


def slugify(text: str) -> str:
    """ASCII slug for episode/reel folder names."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")
    return slug or "untitled"


def map_character_id(character_id: str) -> str:
    return LEGACY_CHARACTER_MAP.get(character_id, character_id)


def map_set_id(location: str, *, beat_title: str = "") -> str:
    """Resolve a beat location string to an assets_3d set ID."""
    if location in {"gingerbread_house", "crone_farm_exterior", "crone_farm_interior"}:
        mapped = LEGACY_SET_MAP.get(location, location)
        interior = _set_from_title(beat_title) == "gingerbread_house_interior"
        if location == "gingerbread_house" and interior:
            return "gingerbread_house_interior"
        return mapped

    if not location:
        return _set_from_title(beat_title) or "forest_path"

    mapped = LEGACY_SET_MAP.get(location, location)
    title_override = _set_from_title(beat_title)
    if location in {"wood", "hut"} and title_override:
        return title_override
    return mapped


def _set_from_title(beat_title: str) -> str | None:
    title = slugify(beat_title).replace("_", " ")
    for keywords, set_id in _TITLE_SET_HINTS:
        if any(keyword in title for keyword in keywords):
            return set_id
    return None


def infer_characters(
    declared: tuple[str, ...],
    *,
    beat_title: str,
) -> list[str]:
    """Return catalog character IDs for a beat, inferring from title when needed."""
    chars = [map_character_id(c) for c in declared]
    title = slugify(beat_title).replace("_", " ")
    if "vater" in title and "father" not in chars:
        chars.append("father")
    if "stiefmutter" in title and "stepmother" not in chars:
        chars.append("stepmother")
    return sorted(set(chars))
