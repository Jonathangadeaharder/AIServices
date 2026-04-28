from typing import Any

from pydantic import BaseModel, Field


class Dialogue(BaseModel):
    """A single line of dialogue."""
    speaker: str
    text: str
    emotion: str | None = None
    tone: str | None = None
    effect: str | None = None


class Shot(BaseModel):
    """A single camera shot within a scene."""
    shot_id: str
    description: str
    dialogue: list[Dialogue] = Field(default_factory=list)
    visual_prompt: str | None = None
    duration_sec: float | None = None


class Scene(BaseModel):
    """A scene containing multiple shots."""
    scene_id: str
    title: str
    location: str
    description: str
    shots: list[Shot] = Field(default_factory=list)


class Episode(BaseModel):
    """A full episode containing multiple scenes."""
    title: str
    cast: dict[str, dict[str, Any]] = Field(default_factory=dict)
    scenes: list[Scene] = Field(default_factory=list)
