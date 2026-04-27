from aiservices_core.providers import registry

from .comfyui import ComfyUIProvider
from .replicate_cloud import ReplicateProvider

registry.register("image2image.comfyui", ComfyUIProvider)
registry.register("image2image.replicate", ReplicateProvider)
