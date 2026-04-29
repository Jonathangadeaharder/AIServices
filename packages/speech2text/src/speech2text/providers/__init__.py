from aiservices_core.providers import registry

from .mlx import MLXWhisperProvider

registry.register("speech2text.mlx", MLXWhisperProvider)
