from aiservices_core.providers import registry

from .elevenlabs import ElevenLabsProvider
from .fish import FishSpeechProvider
from .fish_mlx import FishMLXProvider

registry.register("text2speech.fish", FishSpeechProvider)
registry.register("text2speech.fish_mlx", FishMLXProvider)
registry.register("text2speech.elevenlabs", ElevenLabsProvider)
