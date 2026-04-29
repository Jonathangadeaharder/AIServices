from aiservices_core.providers import registry

from .fish import FishSpeechProvider

registry.register("text2speech.fish", FishSpeechProvider)
