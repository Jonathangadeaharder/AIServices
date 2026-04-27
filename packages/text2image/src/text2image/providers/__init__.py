from aiservices_core.providers import registry

from .comfyui import ComfyUIProvider
from .replicate_cloud import ReplicateProvider

registry.register("comfyui", ComfyUIProvider)
registry.register("replicate", ReplicateProvider)
