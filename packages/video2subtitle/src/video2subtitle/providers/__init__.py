from aiservices_core.providers import registry

from .mlx import MLXProvider

registry.register("video2subtitle.mlx", MLXProvider)
