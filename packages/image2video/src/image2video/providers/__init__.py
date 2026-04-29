from aiservices_core.providers import registry

from .mlx import MLXProvider

registry.register("image2video.mlx", MLXProvider)
