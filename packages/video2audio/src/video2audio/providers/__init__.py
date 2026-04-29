from aiservices_core.providers import registry

from .ffmpeg import FFmpegProvider

registry.register("video2audio.ffmpeg", FFmpegProvider)
