import logging
import os
from io import BytesIO
from pathlib import Path

import pytest


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


def test_save_and_load_image(tmp_path):
    from aiservices_core.io import load_image, save_image
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    out = tmp_path / "test.png"
    save_image(img, out)
    assert out.exists()

    loaded = load_image(out)
    assert loaded.size == (64, 64)


def test_load_image_converts_to_rgb(tmp_path):
    from aiservices_core.io import load_image, save_image
    from PIL import Image

    img = Image.new("RGBA", (32, 32), color=(255, 0, 0, 128))
    out = tmp_path / "rgba.png"
    save_image(img, out)
    loaded = load_image(out)
    assert loaded.mode == "RGB"


def test_load_image_from_url_converts_to_rgb(mocker):
    from aiservices_core.io import load_image
    from PIL import Image

    img = Image.new("RGBA", (32, 32), color=(0, 255, 0, 64))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    mock_resp = mocker.MagicMock()
    mock_resp.content = buf.getvalue()
    mock_resp.raise_for_status = mocker.MagicMock()
    mocker.patch("requests.get", return_value=mock_resp)

    loaded = load_image("https://example.com/image.png")
    assert loaded.mode == "RGB"


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
    from aiservices_core.providers import ProviderRegistry

    reg = ProviderRegistry()
    with pytest.raises(ValueError, match="not found") as exc_info:
        reg.get("nonexistent")
    assert "not found" in str(exc_info.value)


def test_provider_registry_error_lists_available():
    from aiservices_core.providers import BaseProvider, ProviderRegistry

    class StubProvider(BaseProvider):
        def __init__(self, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            return None

    reg = ProviderRegistry()
    reg.register("alpha", StubProvider)
    with pytest.raises(ValueError, match="alpha") as exc_info:
        reg.get("missing")
    assert "alpha" in str(exc_info.value)


def test_get_optimal_device_explicit(monkeypatch, mocker):
    monkeypatch.setenv("AIS_DEVICE", "cpu")
    from aiservices_core.config import AIServicesConfig

    mocker.patch("aiservices_core.runtime.config", AIServicesConfig())
    from aiservices_core.runtime import get_optimal_device

    assert get_optimal_device() == "cpu"


def test_get_optimal_device_auto_darwin_arm(monkeypatch, mocker):
    monkeypatch.setenv("AIS_DEVICE", "auto")
    from aiservices_core.config import AIServicesConfig

    mocker.patch("aiservices_core.runtime.config", AIServicesConfig())
    mocker.patch("aiservices_core.runtime.platform.system", return_value="Darwin")
    mocker.patch("aiservices_core.runtime.platform.machine", return_value="arm64")
    from aiservices_core.runtime import get_optimal_device

    assert get_optimal_device() == "mps"


def test_setup_cache_creates_dir(tmp_path, mocker):
    cache_path = tmp_path / "cache"

    mock_cfg = mocker.patch("aiservices_core.runtime.config")
    mock_cfg.cache_dir = cache_path
    from aiservices_core.runtime import setup_cache

    setup_cache()

    assert cache_path.exists()


def test_setup_cache_idempotent(tmp_path, mocker):
    cache_path = tmp_path / "cache"

    mock_cfg = mocker.patch("aiservices_core.runtime.config")
    mock_cfg.cache_dir = cache_path
    from aiservices_core.runtime import setup_cache

    setup_cache()
    setup_cache()

    assert cache_path.exists()


def test_setup_cache_nested_dir(tmp_path, mocker):
    nested = tmp_path / "deep" / "nested" / "cache"
    mock_cfg = mocker.patch("aiservices_core.runtime.config")
    mock_cfg.cache_dir = nested
    from aiservices_core.runtime import setup_cache

    setup_cache()

    assert nested.exists()


def test_get_logger_returns_logger():
    from aiservices_core.logging import get_logger

    logger = get_logger("test.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.module"


def test_create_progress_bar():
    from aiservices_core.logging import create_progress_bar
    from rich.progress import Progress

    bar = create_progress_bar()
    assert isinstance(bar, Progress)


def test_create_progress_bar_columns():
    from aiservices_core.logging import create_progress_bar
    from rich.progress import BarColumn, SpinnerColumn, TextColumn, TimeElapsedColumn

    bar = create_progress_bar()
    assert len(bar.columns) == 5, "progress bar should have 5 columns"
    assert isinstance(bar.columns[0], SpinnerColumn), "column 0 should be SpinnerColumn"
    assert isinstance(bar.columns[1], TextColumn), "column 1 should be TextColumn"
    assert isinstance(bar.columns[2], BarColumn), "column 2 should be BarColumn"
    assert isinstance(bar.columns[3], TextColumn), "column 3 should be TextColumn"
    assert isinstance(bar.columns[4], TimeElapsedColumn), "column 4 should be TimeElapsedColumn"


def test_create_progress_bar_console():
    from aiservices_core.logging import console, create_progress_bar

    bar = create_progress_bar()
    assert bar.console is console


def test_setup_logging_debug_true(isolated_root_logger):
    from aiservices_core.logging import setup_logging

    isolated_root_logger.handlers.clear()
    setup_logging(debug=True)
    assert isolated_root_logger.level == logging.DEBUG


def test_setup_logging_debug_false(isolated_root_logger):
    from aiservices_core.logging import setup_logging

    isolated_root_logger.handlers.clear()
    setup_logging(debug=False)
    assert isolated_root_logger.level == logging.INFO


def test_setup_logging_format(isolated_root_logger):
    from aiservices_core.logging import setup_logging

    isolated_root_logger.handlers.clear()
    setup_logging(debug=False)
    handler = isolated_root_logger.handlers[-1]
    assert handler.formatter
    assert handler.formatter._fmt == "%(message)s"
    assert handler.formatter.datefmt == "[%X]"


def test_setup_logging_rich_handler(isolated_root_logger):
    from aiservices_core.logging import setup_logging
    from rich.logging import RichHandler

    isolated_root_logger.handlers.clear()
    setup_logging(debug=False)
    handler = isolated_root_logger.handlers[-1]
    assert isinstance(handler, RichHandler)
    assert handler.rich_tracebacks is True


def test_setup_logging_handler_console(isolated_root_logger):
    from aiservices_core.logging import console, setup_logging

    isolated_root_logger.handlers.clear()
    setup_logging(debug=False)
    handler = isolated_root_logger.handlers[-1]
    assert handler.console is console


def test_setup_logging_no_exception(isolated_root_logger):
    from aiservices_core.logging import setup_logging

    isolated_root_logger.handlers.clear()
    setup_logging(debug=False)
    setup_logging(debug=True)
    assert isolated_root_logger.handlers, "root logger should remain configured"


def test_verbose_option_exists():
    import typer
    from aiservices_core.cli import verbose_option

    assert isinstance(verbose_option, typer.models.OptionInfo)


def test_device_option_exists():
    import typer
    from aiservices_core.cli import device_option

    assert isinstance(device_option, typer.models.OptionInfo)


def test_error_hierarchy():
    from aiservices_core.errors import AIServicesError, ProviderError, ResourceNotFoundError

    assert issubclass(ProviderError, AIServicesError)
    assert issubclass(ResourceNotFoundError, AIServicesError)
    assert issubclass(AIServicesError, Exception)


def test_retry_api_call_decorator():
    from aiservices_core.errors import retry_api_call

    call_count = 0

    @retry_api_call
    def flaky():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("transient")

    with pytest.raises(RuntimeError):
        flaky()

    assert call_count == 3


def test_get_optimal_device_auto_linux_returns_cpu(monkeypatch, mocker):
    monkeypatch.setenv("AIS_DEVICE", "auto")
    from aiservices_core.config import AIServicesConfig

    mocker.patch("aiservices_core.runtime.config", AIServicesConfig())
    mocker.patch("aiservices_core.runtime.platform.system", return_value="Linux")
    mocker.patch("aiservices_core.runtime.platform.machine", return_value="x86_64")
    from aiservices_core.runtime import get_optimal_device

    assert get_optimal_device() == "cpu"


def test_get_optimal_device_auto_darwin_x86_returns_cpu(monkeypatch, mocker):
    monkeypatch.setenv("AIS_DEVICE", "auto")
    from aiservices_core.config import AIServicesConfig

    mocker.patch("aiservices_core.runtime.config", AIServicesConfig())
    mocker.patch("aiservices_core.runtime.platform.system", return_value="Darwin")
    mocker.patch("aiservices_core.runtime.platform.machine", return_value="x86_64")
    from aiservices_core.runtime import get_optimal_device

    assert get_optimal_device() == "cpu"


def test_config_auto_cleanup_defaults():
    from aiservices_core.config import AIServicesConfig

    cfg = AIServicesConfig()
    assert cfg.auto_cleanup is True
    assert cfg.max_memory_fraction == 0.8


def test_cleanup_memory(mocker):
    import sys
    from aiservices_core.runtime import cleanup_memory

    # Mock reap_orphaned_processes
    mock_reap = mocker.patch("aiservices_core.runtime.reap_orphaned_processes")

    # Mock torch.mps
    mock_torch = mocker.MagicMock()
    mock_torch.mps.is_available.return_value = True
    mocker.patch.dict(sys.modules, {"torch": mock_torch})

    # Mock mlx.core and parent mlx
    mock_mlx_core = mocker.MagicMock()
    mock_mlx = mocker.MagicMock()
    mock_mlx.core = mock_mlx_core
    mocker.patch.dict(sys.modules, {"mlx.core": mock_mlx_core, "mlx": mock_mlx})

    # Mock gc.collect
    mock_gc = mocker.patch("gc.collect")

    # Mock subprocess.run for system purge
    mock_run = mocker.patch("subprocess.run")
    mocker.patch("aiservices_core.runtime.platform.system", return_value="Darwin")

    cleanup_memory(force_sys_purge=True)

    mock_reap.assert_called_once()
    mock_torch.mps.empty_cache.assert_called_once()
    mock_mlx_core.metal.clear_cache.assert_called_once()
    mock_gc.assert_called_once()
    mock_run.assert_called_once_with(["sudo", "-n", "purge"], capture_output=True, check=False)


def test_setup_memory_guards(mocker):
    import sys
    from aiservices_core.runtime import setup_memory_guards
    from aiservices_core.config import AIServicesConfig

    cfg = AIServicesConfig(max_memory_fraction=0.7)
    mocker.patch("aiservices_core.runtime.config", cfg)
    mocker.patch("aiservices_core.runtime.get_total_memory", return_value=16 * 1024 * 1024 * 1024)

    # 1. Test Modern MLX API (set_memory_limit/set_cache_limit directly on mx/mlx.core)
    mock_torch = mocker.MagicMock()
    mock_torch.mps.is_available.return_value = True
    mocker.patch.dict(sys.modules, {"torch": mock_torch})

    mock_mlx_core = mocker.MagicMock()
    mock_mlx = mocker.MagicMock()
    mock_mlx.core = mock_mlx_core
    mocker.patch.dict(sys.modules, {"mlx": mock_mlx, "mlx.core": mock_mlx_core})

    setup_memory_guards()

    mock_torch.mps.set_per_process_memory_fraction.assert_called_once_with(0.7)
    expected_limit = int(16 * 1024 * 1024 * 1024 * 0.7)
    mock_mlx_core.set_memory_limit.assert_called_once_with(expected_limit)
    mock_mlx_core.set_cache_limit.assert_called_once_with(int(expected_limit * 0.1))
    mock_mlx_core.metal.set_memory_limit.assert_not_called()

    # 2. Test Legacy MLX API (set_memory_limit/set_cache_limit on mx.metal)
    mock_torch.reset_mock()
    mock_mlx_core_legacy = mocker.MagicMock(spec=["metal"])
    # Delete the attribute to simulate lack of modern API
    del mock_mlx_core_legacy.set_memory_limit
    del mock_mlx_core_legacy.set_cache_limit
    
    mock_mlx_legacy = mocker.MagicMock()
    mock_mlx_legacy.core = mock_mlx_core_legacy
    mocker.patch.dict(sys.modules, {"mlx": mock_mlx_legacy, "mlx.core": mock_mlx_core_legacy})

    setup_memory_guards()

    mock_torch.mps.set_per_process_memory_fraction.assert_called_once_with(0.7)
    mock_mlx_core_legacy.metal.set_memory_limit.assert_called_once_with(expected_limit)
    mock_mlx_core_legacy.metal.set_cache_limit.assert_called_once_with(int(expected_limit * 0.1))


def test_memory_managed_provider_decorator(mocker):
    from aiservices_core.providers import BaseProvider, MemoryManagedProvider, ProviderRegistry
    from aiservices_core.config import AIServicesConfig

    cfg = AIServicesConfig(auto_cleanup=True)
    mocker.patch("aiservices_core.providers.config", cfg)

    mock_setup_guards = mocker.patch("aiservices_core.runtime.setup_memory_guards")
    mock_cleanup = mocker.patch("aiservices_core.runtime.cleanup_memory")

    class TestProvider(BaseProvider):
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.called_generate = False
            self.some_attr = "hello"

        def generate(self, request, output_path=None):
            self.called_generate = True
            return "success"

    # Verify decorator wraps on ProviderRegistry.get
    reg = ProviderRegistry()
    reg.register("test", TestProvider)

    provider = reg.get("test", foo="bar")
    assert isinstance(provider, MemoryManagedProvider)
    # Verify isinstance also works for the wrapped class
    assert isinstance(provider, TestProvider)

    # Verify memory guards are set up on init
    mock_setup_guards.assert_called_once()

    # Verify attribute delegation
    assert provider.some_attr == "hello"
    assert provider.kwargs == {"foo": "bar"}

    # Verify generate calls cleanup
    res = provider.generate("req", "out")
    assert res == "success"
    assert provider._provider.called_generate is True
    mock_cleanup.assert_called_once()


