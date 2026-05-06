from aiservices_core.providers import registry

from .fish_mlx import FishMLXProvider

registry.register("text2audio.fish_mlx", FishMLXProvider)
