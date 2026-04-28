"""Tests for aiservices_core package modules."""
import os
from pathlib import Path
from unittest.mock import patch

# ── E0.3: config ──────────────────────────────────────────────────────────────


def test_version():
    from aiservices_core import __version__

    assert __version__ == "0.1.0"


def test_config_defaults():
    from aiservices_core.config import AIServicesConfig

    cfg = AIServicesConfig()
    assert cfg.cache_dir == Path(os.path.expanduser("~/.cache/aiservices"))
    assert cfg.device == "auto"
    assert cfg.debug is False


def test_config_env_prefix(monkeypatch):
    monkeypatch.setenv("AIS_DEBUG", "true")
    monkeypatch.setenv("AIS_DEVICE", "cpu")
    from aiservices_core.config import AIServicesConfig

    cfg = AIServicesConfig()
    assert cfg.debug is True
    assert cfg.device == "cpu"


# ── E0.4: io ──────────────────────────────────────────────────────────────────


def test_save_and_load_image(tmp_path):
    from aiservices_core.io import load_image, save_image
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    out = tmp_path / "test.png"
    save_image(img, out)
    assert out.exists()

    loaded = load_image(out)
    assert loaded.size == (64, 64)


def test_save_image_creates_parent_dirs(tmp_path):
    from aiservices_core.io import save_image
    from PIL import Image

    img = Image.new("RGB", (32, 32))
    nested = tmp_path / "a" / "b" / "c" / "out.png"
    save_image(img, nested)
    assert nested.exists()


# ── E0.5: providers ───────────────────────────────────────────────────────────


def test_provider_registry_register_and_get():
    from aiservices_core.providers import BaseProvider, ProviderRegistry

    class DummyProvider(BaseProvider):
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, *args, **kwargs):
            return None

    reg = ProviderRegistry()
    reg.register("dummy", DummyProvider)
    instance = reg.get("dummy", foo="bar")
    assert isinstance(instance, DummyProvider)
    assert instance.kwargs.get("foo") == "bar"


def test_provider_registry_unknown_raises():
    import pytest
    from aiservices_core.providers import ProviderRegistry

    reg = ProviderRegistry()
    with pytest.raises(ValueError, match="not found"):
        reg.get("nonexistent")


# ── E0.6: runtime ─────────────────────────────────────────────────────────────


def test_get_optimal_device_explicit(monkeypatch):
    monkeypatch.setenv("AIS_DEVICE", "cpu")
    from aiservices_core import config as cfg_module
    from aiservices_core.config import AIServicesConfig

    cfg_module.config = AIServicesConfig()
    from aiservices_core.runtime import get_optimal_device

    assert get_optimal_device() == "cpu"


def test_get_optimal_device_auto_darwin_arm(monkeypatch):
    monkeypatch.setenv("AIS_DEVICE", "auto")
    from aiservices_core import config as cfg_module
    from aiservices_core.config import AIServicesConfig

    cfg_module.config = AIServicesConfig()
    with patch("platform.system", return_value="Darwin"), patch(
        "platform.machine", return_value="arm64"
    ):
        from importlib import reload

        from aiservices_core import runtime as rt_module

        reload(rt_module)
        assert rt_module.get_optimal_device() == "mps"


def test_setup_cache_creates_dir(tmp_path):
    """setup_cache creates the cache directory defined in config."""
    from unittest.mock import patch

    cache_path = tmp_path / "cache"

    # Patch the config.cache_dir that runtime.py references at call-time
    with patch("aiservices_core.runtime.config") as mock_cfg:
        mock_cfg.cache_dir = cache_path
        from aiservices_core.runtime import setup_cache

        setup_cache()

    assert cache_path.exists()


# ── E0.7: logging ─────────────────────────────────────────────────────────────


def test_get_logger_returns_logger():
    import logging

    from aiservices_core.logging import get_logger

    logger = get_logger("test.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.module"


def test_create_progress_bar():
    from aiservices_core.logging import create_progress_bar
    from rich.progress import Progress

    bar = create_progress_bar()
    assert isinstance(bar, Progress)


def test_setup_logging_no_exception():
    from aiservices_core.logging import setup_logging

    setup_logging(debug=False)
    setup_logging(debug=True)


# ── E0.8: cli ─────────────────────────────────────────────────────────────────


def test_verbose_option_exists():
    import typer
    from aiservices_core.cli import verbose_option

    assert isinstance(verbose_option, typer.models.OptionInfo)


def test_device_option_exists():
    import typer
    from aiservices_core.cli import device_option

    assert isinstance(device_option, typer.models.OptionInfo)


# ── E0.9: errors ──────────────────────────────────────────────────────────────


def test_error_hierarchy():
    from aiservices_core.errors import AIServicesError, ProviderError, ResourceNotFoundError

    assert issubclass(ProviderError, AIServicesError)
    assert issubclass(ResourceNotFoundError, AIServicesError)
    assert issubclass(AIServicesError, Exception)


def test_retry_api_call_decorator():
    """retry_api_call should retry transient failures and eventually raise."""
    import pytest
    from aiservices_core.errors import retry_api_call

    call_count = 0

    @retry_api_call
    def flaky():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("transient")

    with pytest.raises(RuntimeError):
        flaky()

    assert call_count == 3  # stop_after_attempt(3)
