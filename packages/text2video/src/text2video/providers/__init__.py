from aiservices_core.providers import registry

from .mlx import MLXProvider

registry.register("text2video.mlx", MLXProvider)
