"""Build chapter narration audio from episode shot lists."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# TODO: import from scriptforge
# from forge.audio_gen.assemble import assemble_narration
# from forge.config.models import TTS_SAMPLE_RATE
# from forge.domain.artifact import AudioClip
# from forge.domain.beat import Beat
from blenderservices.scaffold import load_shot_list_for_chapter

# TODO: import from scriptforge — placeholders until scriptforge provides these
TTS_SAMPLE_RATE = 22050  # matches forge.config.models.TTS_SAMPLE_RATE

class AudioClip:
    """Placeholder until scriptforge provides forge.domain.artifact.AudioClip."""
    def __init__(self, path, duration_s, sample_rate):
        self.path = path
        self.duration_s = duration_s
        self.sample_rate = sample_rate

class Beat:
    """Placeholder until scriptforge provides forge.domain.beat.Beat."""
    def __init__(self, *, index, chapter, narrator, description, characters, location):
        self.index = index
        self.chapter = chapter
        self.narrator = narrator
        self.description = description
        self.characters = characters
        self.location = location

def assemble_narration(clips, out_path):
    """Placeholder until scriptforge provides forge.audio_gen.assemble.assemble_narration."""
    import numpy as np
    import soundfile as sf
    segments = []
    for clip in clips:
        data, sr = sf.read(str(clip.path), dtype="float32")
        segments.append(data)
    if segments:
        combined = np.concatenate(segments)
    else:
        combined = np.zeros(0, dtype="float32")
    sf.write(str(out_path), combined, TTS_SAMPLE_RATE)


def _strip_narrator_tags(text: str) -> str:
    clean = re.sub(r"\[.*?\]", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _silence_clip(
    out_path: Path, duration_sec: float, sample_rate: int = TTS_SAMPLE_RATE
) -> AudioClip:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples = max(int(duration_sec * sample_rate), 1)
    sf.write(out_path, np.zeros(samples, dtype=np.float32), sample_rate)
    return AudioClip(path=out_path, duration_s=duration_sec, sample_rate=sample_rate)


def _try_tts_clip(
    beat: Beat,
    out_path: Path,
    *,
    anchor_path: Path | None,
    anchor_text: str,
    duration_sec: float,
) -> AudioClip | None:
    if anchor_path is not None and anchor_path.is_file():
        try:
            # TODO: import from scriptforge
            # from forge.audio_gen.tts import generate_clip
            # return generate_clip(beat, anchor_path, anchor_text, out_path)
            pass
        except Exception:
            pass

    spoken = _strip_narrator_tags(beat.narrator)
    if spoken and shutil.which("say") is not None:
        return _macos_say_clip(spoken, out_path, duration_sec)

    return None


def _macos_say_clip(text: str, out_path: Path, target_duration: float | None) -> AudioClip | None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "clip.aiff"
        result = subprocess.run(
            ["say", "-o", str(aiff), text],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not aiff.is_file():
            return None

        data, rate = sf.read(str(aiff), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        if target_duration and target_duration > 0:
            target_samples = int(target_duration * rate)
            if len(data) < target_samples:
                pad = np.zeros(target_samples - len(data), dtype=np.float32)
                data = np.concatenate([data, pad])
            elif len(data) > target_samples:
                data = data[:target_samples]

        sf.write(str(out_path), data, rate)
    duration_s = len(data) / rate
    return AudioClip(path=out_path, duration_s=duration_s, sample_rate=rate)


def build_chapter_narration(
    project_root: Path,
    chapter_num: int,
    output_path: Path | None = None,
    *,
    anchor_path: Path | None = None,
    anchor_text: str = "",
    use_tts: bool = True,
) -> Path:
    """Merge per-shot narrator lines into one chapter WAV (TTS or timed silence)."""
    doc = load_shot_list_for_chapter(project_root, chapter_num)
    audio_dir = project_root / "build" / "renders" / doc.episode / "narration"
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_path or audio_dir / f"chapter_{chapter_num:02d}.wav"

    clips: list[AudioClip] = []
    for index, plan in enumerate(doc.shots, start=1):
        clip_path = audio_dir / f"{plan.id}.wav"
        beat = Beat(
            index=index,
            chapter=chapter_num,
            narrator=plan.narrator,
            description=plan.notes,
            characters=tuple(plan.character_ids),
            location=plan.set_id,
        )
        clip: AudioClip | None = None
        if use_tts and plan.narrator.strip():
            clip = _try_tts_clip(
                beat,
                clip_path,
                anchor_path=anchor_path,
                anchor_text=anchor_text,
                duration_sec=float(plan.duration_sec),
            )

        if clip is None:
            clip = _silence_clip(clip_path, float(plan.duration_sec))

        clips.append(clip)

    assemble_narration(clips, out_path)
    return out_path
