from aiservices_core.providers import registry

from .comfyui import ComfyUIProvider
from .replicate_cloud import ReplicateProvider

registry.register("text2image.comfyui", ComfyUIProvider)
registry.register("text2image.replicate", ReplicateProvider)
