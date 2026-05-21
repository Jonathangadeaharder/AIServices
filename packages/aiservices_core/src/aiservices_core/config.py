import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AIServicesConfig(BaseSettings):
    """Base configuration for AIServices modules."""

    # Common settings
    cache_dir: Path = Path(os.path.expanduser("~/.cache/aiservices"))
    device: str = "auto"
    debug: bool = False
    auto_cleanup: bool = True
    max_memory_fraction: float = 0.8

    model_config = SettingsConfigDict(env_prefix="AIS_")


config = AIServicesConfig()
