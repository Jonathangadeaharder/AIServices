"""Generate Studio shot lists from story beats (Phase 4 production route)."""

from __future__ import annotations

import importlib
import sys
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

# TODO: import from scriptforge
# from forge.domain.story import StoryBeat, StoryChapter, StorySpec, load_spec
from blenderservices.catalog import PRODUCTION_CATALOG
from blenderservices.location_map import infer_characters, map_set_id, slugify
from blenderservices.manifest import AssetStatus, AssetType, ShotManifest

# TODO: import from scriptforge — placeholders until scriptforge provides these
class StoryBeat:
    """Placeholder until scriptforge provides forge.domain.story.StoryBeat."""
    def __init__(self, *, num: int | str, title: str, duration: int = 20,
                 characters: tuple[str, ...] = (), location: str = "",
                 narrator: str = "", ember: bool = False):
        self.num = int(num) if isinstance(num, str) else num
        self.title = title
        self.duration = duration
        self.characters = characters
        self.location = location
        self.narrator = narrator
        self.ember = ember

class StoryChapter:
    """Placeholder until scriptforge provides forge.domain.story.StoryChapter."""
    def __init__(self, *, num: int | str, title: str, beats: list[StoryBeat]):
        self.num = int(num) if isinstance(num, str) else num
        self.title = title
        self.beats = beats

class StorySpec:
    """Placeholder until scriptforge provides forge.domain.story.StorySpec."""
    def __init__(self, *, chapters: list[StoryChapter]):
        self.chapters = chapters


def load_spec(project_root: Path) -> StorySpec:
    """Load story spec from .project.md YAML frontmatter or legacy project.forge.yaml."""
    project_md = project_root / ".project.md"
    forge_yaml = project_root / "project.forge.yaml"

    if project_md.is_file():
        return _load_spec_from_project_md(project_md)
    if forge_yaml.is_file():
        return _load_spec_from_forge_yaml(forge_yaml)

    raise FileNotFoundError(f"no .project.md or project.forge.yaml in {project_root}")


def _load_spec_from_project_md(path: Path) -> StorySpec:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path} must start with YAML frontmatter")
    end = text.find("---", 3)
    if end == -1:
        raise ValueError(f"{path}: unclosed YAML frontmatter")
    frontmatter = yaml.safe_load(text[3:end])
    chapters = []
    for beat in frontmatter.get("beats", []):
        ch_num = int(beat.get("chapter", 1))
        while len(chapters) < ch_num:
            chapters.append(StoryChapter(num=len(chapters) + 1, title=f"Chapter {len(chapters) + 1}", beats=[]))
        chapters[ch_num - 1].beats.append(StoryBeat(
            num=int(beat.get("id", "").split("_")[-1]) if "_" in str(beat.get("id", "")) else len(chapters[ch_num - 1].beats) + 1,
            title=beat.get("narrator", ""),
            duration=int(beat.get("duration", 20)),
            characters=tuple(beat.get("characters", [])),
            location=beat.get("location", ""),
            narrator=beat.get("narrator", ""),
            ember=False,
        ))
    return StorySpec(chapters=chapters)


def _load_spec_from_forge_yaml(path: Path) -> StorySpec:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    chapters = []
    project_root = path.parent
    for ch_entry in data.get("chapters", []):
        source = ch_entry.get("source", "")
        ch_num = int(ch_entry.get("num", "1"))
        ch_title = ch_entry.get("title", f"Chapter {ch_num}")
        beats_file = project_root / source if source else project_root / "chapters" / f"ch{ch_num:02d}_beats.yaml"
        beats = []
        if beats_file.is_file():
            beats_data = yaml.safe_load(beats_file.read_text(encoding="utf-8"))
            for b in beats_data.get("beats", []):
                beats.append(StoryBeat(
                    num=int(b.get("num", 1)),
                    title=b.get("title", ""),
                    duration=int(b.get("duration", 20)),
                    characters=tuple(b.get("characters", [])),
                    location=b.get("location", ""),
                    narrator=b.get("narrator", ""),
                    ember=b.get("ember", False),
                ))
        chapters.append(StoryChapter(num=ch_num, title=ch_title, beats=beats))
    return StorySpec(chapters=chapters)

CatalogIds = frozenset(asset.id for asset in PRODUCTION_CATALOG)


class ShotGrouping(StrEnum):
    SINGLE = "single"
    COMBINED = "combined"
    MONTAGE = "montage"


class EpisodeShotPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    beat_ids: list[str]
    grouping: ShotGrouping = ShotGrouping.SINGLE
    set_id: str
    character_ids: list[str] = Field(default_factory=list)
    prop_ids: list[str] = Field(default_factory=list)
    action_library_ids: list[str] = Field(default_factory=list)
    character_actions: dict[str, str] = Field(default_factory=dict)
    camera: str | None = None
    lighting: str | None = None
    duration_sec: int = 20
    frames: int = 240
    narrator: str = ""
    notes: str = ""


class ShotListDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    chapter: int
    episode: str
    reel: str
    title: str
    shots: list[EpisodeShotPlan]


def beat_id(chapter_num: str, beat_num: str) -> str:
    return f"ch{chapter_num}_beat_{beat_num}"


def frames_for_duration(duration_sec: int, *, fps: int = 24) -> int:
    return max(fps * 3, duration_sec * fps)


def infer_props(beat: StoryBeat) -> list[str]:
    title = beat.title.lower()
    props: list[str] = []
    if any(word in title for word in ("kiesel", "stein")):
        props.append("stones")
    if any(word in title for word in ("suppe", "schussel", "schüssel")):
        props.append("bowl")
    return [prop for prop in props if prop in CatalogIds]


def infer_actions(
    *,
    set_id: str,
    character_ids: list[str],
    beat: StoryBeat,
) -> tuple[list[str], dict[str, str]]:
    if not character_ids:
        return [], {}

    title = beat.title.lower()
    action_library_ids: list[str] = []
    character_actions: dict[str, str] = {}

    if "shared_locomotion" in CatalogIds:
        action_library_ids.append("shared_locomotion")

    motion = "idle"
    if set_id in {"forest_path", "forest_edge", "deep_forest", "river_crossing"}:
        motion = "walk"
    if any(word in title for word in ("lauf", "aufbruch", "pfad", "schwelle")):
        motion = "walk"
    if any(word in title for word in ("weint", "hunger", "hor", "horcht", "stille")):
        motion = "tired_walk" if motion == "walk" else "listen"
    if "angst" in title or "furcht" in title:
        motion = "fear"

    for char_id in character_ids:
        character_actions[char_id] = motion

    return action_library_ids, character_actions


def infer_camera(*, character_ids: list[str], beat: StoryBeat) -> str:
    title = beat.title.lower()
    if not character_ids:
        return "cam_wide"
    if any(word in title for word in ("hand", "schwelle", "tur", "stimme")):
        return "cam_hero"
    if len(character_ids) >= 2 or "dialog" in title:  # noqa: PLR2004
        return "cam_medium"
    return "cam_medium"


def infer_lighting(*, set_id: str, beat: StoryBeat) -> str:
    title = slugify(beat.title).replace("_", " ")
    if beat.ember or set_id in {"oven_room", "gingerbread_house_interior"}:
        return "preset_ember"
    if set_id in {"deep_forest", "root_shelter", "cellar"}:
        return "preset_moon"
    if set_id == "hut_exterior":
        return "preset_dusk_sun"
    if any(word in title for word in ("nacht", "stroh", "mond")):
        return "preset_moon"
    return "preset_noon_sun"


def plan_shot(
    chapter: StoryChapter,
    beat: StoryBeat,
    *,
    narrator: str = "",
) -> EpisodeShotPlan:
    ch_num = str(chapter.num).zfill(2)
    shot_num = str(beat.num).zfill(2)
    shot_id = f"ep{ch_num}_reel{ch_num}_shot{shot_num}"
    set_id = map_set_id(beat.location, beat_title=beat.title)
    character_ids = infer_characters(beat.characters, beat_title=beat.title)
    prop_ids = infer_props(beat)
    action_library_ids, character_actions = infer_actions(
        set_id=set_id,
        character_ids=character_ids,
        beat=beat,
    )

    return EpisodeShotPlan(
        id=shot_id,
        beat_ids=[beat_id(ch_num, str(beat.num).zfill(2))],
        grouping=ShotGrouping.SINGLE,
        set_id=set_id,
        character_ids=character_ids,
        prop_ids=prop_ids,
        action_library_ids=action_library_ids,
        character_actions=character_actions,
        camera=infer_camera(character_ids=character_ids, beat=beat),
        lighting=infer_lighting(set_id=set_id, beat=beat),
        duration_sec=beat.duration,
        frames=frames_for_duration(beat.duration),
        narrator=narrator,
        notes=beat.title,
    )


def generate_shot_list(
    spec: StorySpec,
    chapter_num: int,
    *,
    narrators: dict[str, str] | None = None,
    project_root: Path | None = None,
) -> ShotListDocument:
    chapter = _find_chapter(spec, chapter_num)
    ch_num = str(chapter.num).zfill(2)
    episode = _episode_folder(project_root, ch_num, chapter.title)
    reel = _reel_folder(project_root, episode, ch_num, chapter.title)
    narrators = narrators or {}

    shots: list[EpisodeShotPlan] = []
    for beat in chapter.beats:
        b_id = beat_id(ch_num, str(beat.num).zfill(2))
        shots.append(
            plan_shot(chapter, beat, narrator=narrators.get(b_id, "")),
        )

    return ShotListDocument(
        chapter=chapter_num,
        episode=episode,
        reel=reel,
        title=chapter.title,
        shots=shots,
    )


def validate_shot_plans(
    doc: ShotListDocument,
    *,
    assets_root: Path = Path("assets_3d"),
) -> list[str]:
    """Return validation errors for catalog references."""
    issues: list[str] = []
    known_sets = _catalog_ids(assets_root, "sets")
    known_chars = _catalog_ids(assets_root, "characters")
    known_props = _catalog_ids(assets_root, "props")
    known_actions = _catalog_ids(assets_root, "actions")

    for shot in doc.shots:
        if shot.set_id not in known_sets:
            issues.append(f"{shot.id}: unknown set_id '{shot.set_id}'")
        for char_id in shot.character_ids:
            if char_id not in known_chars:
                issues.append(f"{shot.id}: unknown character '{char_id}'")
        for prop_id in shot.prop_ids:
            if prop_id not in known_props:
                issues.append(f"{shot.id}: unknown prop '{prop_id}'")
        for action_id in shot.action_library_ids:
            if action_id not in known_actions:
                issues.append(f"{shot.id}: unknown action library '{action_id}'")
    return issues


def write_shot_list(
    doc: ShotListDocument,
    project_root: Path,
    *,
    write_manifests: bool = True,
    write_episode_toml: bool = True,
) -> list[Path]:
    """Write shot_list.yaml and optional per-shot manifests under episodes/."""
    written: list[Path] = []
    episode_dir = project_root / "shot_production" / doc.episode
    episode_dir.mkdir(parents=True, exist_ok=True)

    shot_list_path = episode_dir / "shot_list.yaml"
    shot_list_path.write_text(
        yaml.safe_dump(doc.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    written.append(shot_list_path)

    if write_episode_toml:
        toml_path = _write_episode_toml(doc, episode_dir)
        written.append(toml_path)

    if write_manifests:
        shots_dir = episode_dir / "reels" / doc.reel / "shots"
        shots_dir.mkdir(parents=True, exist_ok=True)
        for shot in doc.shots:
            shot_num = shot.id.rsplit("_", maxsplit=1)[-1]
            blend_name = f"{shot_num}.blend"
            manifest = ShotManifest(
                id=shot.id,
                asset_type=AssetType.SHOT,
                status=AssetStatus.PLANNED,
                blend_file=blend_name,
                preview_files=[],
                license="project-owned",
                notes=shot.notes,
                set_id=shot.set_id,
                character_ids=shot.character_ids,
                prop_ids=shot.prop_ids,
                action_library_ids=shot.action_library_ids,
                character_actions=shot.character_actions,
                camera=shot.camera,
                lighting=shot.lighting,
            )
            manifest_path = shots_dir / f"{shot_num}.manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
                encoding="utf-8",
            )
            written.append(manifest_path)

    return written


def load_narrators_from_scripts(project_root: Path, chapter_num: int) -> dict[str, str]:
    """Load narrator lines keyed by beat_id when scripts/beats modules exist."""
    root = project_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    module_names = (
        f"gretel.content.beats_chapter{chapter_num:02d}",
        f"scripts.beats.beats_chapter{chapter_num:02d}",
    )
    raw_beats: list[dict] = []
    for module_name in module_names:
        try:
            mod = importlib.import_module(module_name)
            raw_beats = getattr(mod, "BEATS", [])
            break
        except (ImportError, AttributeError):
            continue

    ch_num = f"{chapter_num:02d}"
    narrators: dict[str, str] = {}
    for entry in raw_beats:
        beat_no = str(entry.get("beat", entry.get("num", ""))).zfill(2)
        if not beat_no.strip("0"):
            continue
        b_id = beat_id(ch_num, beat_no)
        narrator = str(entry.get("narrator", "") or "")
        if narrator:
            narrators[b_id] = narrator
    return narrators


def generate_chapter_shot_list(
    project_root: Path,
    chapter_num: int,
    *,
    write: bool = False,
    assets_root: Path | None = None,
) -> tuple[ShotListDocument, list[str], list[Path]]:
    """Load spec, build shot list, validate, optionally write artifacts."""
    spec = load_spec(project_root)
    narrators = load_narrators_from_scripts(project_root, chapter_num)
    doc = generate_shot_list(
        spec,
        chapter_num,
        narrators=narrators,
        project_root=project_root,
    )
    issues = validate_shot_plans(doc, assets_root=assets_root or project_root / "assets_3d")
    written: list[Path] = []
    if write and not issues:
        written = write_shot_list(doc, project_root)
    return doc, issues, written


def load_shot_list(path: Path) -> ShotListDocument:
    """Load a shot_list.yaml written by write_shot_list()."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ShotListDocument.model_validate(data)


def _find_chapter(spec: StorySpec, chapter_num: int) -> StoryChapter:
    for chapter in spec.chapters:
        if int(chapter.num) == chapter_num:
            return chapter
    msg = f"chapter {chapter_num} not found in story spec"
    raise ValueError(msg)


def _episode_folder(project_root: Path | None, ch_num: str, title: str) -> str:
    default = f"ep{ch_num}_{slugify(title)}"
    if project_root is None:
        return default
    episodes = project_root / "shot_production"
    if not episodes.is_dir():
        return default
    prefix = f"ep{ch_num}"
    matches = sorted(
        path.name for path in episodes.iterdir() if path.is_dir() and path.name.startswith(prefix)
    )
    return matches[0] if matches else default


def _reel_folder(project_root: Path | None, episode: str, ch_num: str, title: str) -> str:
    default = f"reel{ch_num}_{slugify(title)}"
    if project_root is None:
        return default
    reels = project_root / "shot_production" / episode / "reels"
    if not reels.is_dir():
        return default
    prefix = f"reel{ch_num}"
    matches = sorted(
        path.name for path in reels.iterdir() if path.is_dir() and path.name.startswith(prefix)
    )
    return matches[0] if matches else default


def _catalog_ids(assets_root: Path, kind: str) -> set[str]:
    kind_dir = assets_root / kind
    if not kind_dir.is_dir():
        return set(CatalogIds)
    return {path.name for path in kind_dir.iterdir() if path.is_dir()}


def _write_episode_toml(doc: ShotListDocument, episode_dir: Path) -> Path:
    template = Path(__file__).resolve().parents[3] / "templates" / "_episode" / "episode.toml"
    content = template.read_text(encoding="utf-8") if template.is_file() else _DEFAULT_EPISODE_TOML

    cast = sorted({char for shot in doc.shots for char in shot.character_ids})
    runtime = sum(shot.duration_sec for shot in doc.shots)
    ch_num = str(doc.chapter).zfill(2)
    content = content.replace("TITLE_HERE", doc.title)
    content = content.replace("leads = []", f"leads = {cast}")
    content = content.replace("runtime_sec = 0", f"runtime_sec = {runtime}")
    content = content.replace(f'{ch_num} = ""', f'{ch_num} = "{slugify(doc.title)}"')
    content = content.replace('status      = "not_started"', 'status      = "storyboarded"')

    path = episode_dir / "episode.toml"
    path.write_text(content, encoding="utf-8")
    return path


_DEFAULT_EPISODE_TOML = """\
[meta]
title       = "TITLE_HERE"
source      = "FOLKTALE_SOURCE"
runtime_sec = 0
status      = "not_started"

[cast]
leads = []
antag = []
extra = []

[reels]
01 = ""

[render]
engine     = "EEVEE_NEXT"
resolution = [2304, 960]
fps_source = 24
fps_step   = 3
colorspace = "Filmic"
"""
