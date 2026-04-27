from aiservices_core.providers import registry

from .local_sdxl import LocalSDXLProvider
from .replicate_cloud import ReplicateProvider

registry.register("local_sdxl", LocalSDXLProvider)
registry.register("replicate", ReplicateProvider)
