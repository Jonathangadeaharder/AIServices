from aiservices_core.providers import registry

from .comfyui import ComfyUIProvider
from .mlx import MLXProvider

registry.register("text2video.comfyui", ComfyUIProvider)
registry.register("text2video.mlx", MLXProvider)
