import logging
from pathlib import Path
from typing import Any

import ffmpeg
from text2speech.models import Text2SpeechRequest
from text2speech.providers import registry as tts_registry
from text2video.models import Text2VideoRequest
from text2video.providers import registry as t2v_registry

from ..models import Episode, Scene, Shot

logger = logging.getLogger(__name__)


class Showrunner:
    """Orchestrates the generation of a full episode."""

    def __init__(
        self,
        output_dir: str | Path,
        tts_provider: str = "text2speech.fish",
        t2v_provider: str = "text2video.comfyui",
        device: str = "auto",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tts = tts_registry.get(tts_provider, device=device)
        self.t2v = t2v_registry.get(t2v_provider, device=device)

    def render_episode(self, episode: Episode) -> Path:
        """Render a full episode."""
        logger.info(f"Rendering episode: {episode.title}")
        
        scene_videos = []
        for scene in episode.scenes:
            scene_video = self.render_scene(scene, episode.cast)
            scene_videos.append(scene_video)
            
        # Final assembly
        final_path = self.output_dir / f"{episode.title.replace(' ', '_')}.mp4"
        self.concatenate_videos(scene_videos, final_path)
        
        return final_path

    def render_scene(self, scene: Scene, cast: dict[str, Any]) -> Path:
        """Render all shots in a scene and combine them."""
        logger.info(f"Rendering scene {scene.scene_id}: {scene.title}")
        
        shot_videos = []
        for shot in scene.shots:
            shot_video = self.render_shot(shot, scene, cast)
            shot_videos.append(shot_video)
            
        scene_path = self.output_dir / f"scene_{scene.scene_id}.mp4"
        self.concatenate_videos(shot_videos, scene_path)
        
        return scene_path

    def render_shot(self, shot: Shot, scene: Scene, cast: dict[str, Any]) -> Path:
        """Generate audio and video for a shot and merge them."""
        logger.info(f"Rendering shot {shot.shot_id}")
        
        shot_dir = self.output_dir / "temp" / shot.shot_id
        shot_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Generate Dialogue Audio
        audio_paths = []
        for i, diag in enumerate(shot.dialogue):
            audio_path = shot_dir / f"diag_{i}.wav"
            char_info = cast.get(diag.speaker, {})
            
            request = Text2SpeechRequest(
                text=diag.text,
                voice_id=char_info.get("voice_id"),
                emotion=diag.emotion,
                tone=diag.tone,
                effect=diag.effect
            )
            self.tts.generate(request, str(audio_path))
            audio_paths.append(audio_path)
            
        # Combine dialogue audio
        combined_audio = shot_dir / "dialogue_full.wav"
        if audio_paths:
            self.concatenate_audio(audio_paths, combined_audio)
        
        # 2. Generate Video
        video_path = shot_dir / "visual.mp4"
        t2v_request = Text2VideoRequest(
            prompt=shot.visual_prompt or shot.description,
            num_frames=81, # Default for ~3s
        )
        self.t2v.generate(t2v_request, str(video_path))
        
        # 3. Merge Audio and Video
        output_path = self.output_dir / f"shot_{shot.shot_id}.mp4"
        self.merge_audio_video(video_path, combined_audio, output_path)
        
        return output_path

    def concatenate_audio(self, paths: list[Path], output: Path):
        """Concatenate multiple audio files with a small gap."""
        inputs = [ffmpeg.input(str(p)) for p in paths]
        stream = ffmpeg.concat(*inputs, v=0, a=1).output(str(output))
        ffmpeg.run(stream, overwrite_output=True, quiet=True)

    def concatenate_videos(self, paths: list[Path], output: Path):
        """Concatenate multiple video files."""
        inputs = [ffmpeg.input(str(p)) for p in paths]
        stream = ffmpeg.concat(*inputs).output(str(output))
        ffmpeg.run(stream, overwrite_output=True, quiet=True)

    def merge_audio_video(self, video: Path, audio: Path, output: Path):
        """Merge a video stream and an audio stream."""
        v = ffmpeg.input(str(video))
        if audio.exists():
            a = ffmpeg.input(str(audio))
            stream = ffmpeg.output(v, a, str(output), vcodec="copy", acodec="aac")
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
        else:
            stream = ffmpeg.output(v, str(output), vcodec="copy")
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
