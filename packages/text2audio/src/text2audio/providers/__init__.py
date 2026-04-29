from aiservices_core.providers import registry

from .replicate_cloud import ReplicateProvider

registry.register("text2audio.replicate", ReplicateProvider)
