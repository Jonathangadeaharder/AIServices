from aiservices_core.providers import registry

from .mlx import MLXWhisperProvider

registry.register("audio2subtitle.mlx", MLXWhisperProvider)
