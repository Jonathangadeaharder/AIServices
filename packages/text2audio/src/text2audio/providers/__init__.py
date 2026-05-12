from aiservices_core.providers import registry

from .mlx import MLXProvider

registry.register("text2audio.mlx", MLXProvider)
