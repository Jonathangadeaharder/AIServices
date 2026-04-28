from aiservices_core.providers import registry

from .comfyui import ComfyUIProvider
from .mlx import MLXProvider
from .replicate_cloud import ReplicateProvider

registry.register("image2image.comfyui", ComfyUIProvider)
registry.register("image2image.mlx", MLXProvider)
registry.register("image2image.replicate", ReplicateProvider)
