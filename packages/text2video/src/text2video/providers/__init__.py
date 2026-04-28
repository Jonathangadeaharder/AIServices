from aiservices_core.providers import registry

from .comfyui import ComfyUIProvider

registry.register("text2video.comfyui", ComfyUIProvider)
