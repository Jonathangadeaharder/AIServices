from aiservices_core.providers import registry

from .comfyui import ComfyUIProvider
from .mlx import MLXProvider

registry.register("image2image.clothing", ComfyUIProvider)
registry.register("image2image.mlx", MLXProvider)
