import logging
import os
from io import BytesIO
from pathlib import Path

import pytest


@pytest.fixture
def isolated_root_logger():
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    root.handlers.clear()
    yield root
    root.handlers.clear()
    root.handlers.extend(old_handlers)
    root.setLevel(old_level)


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
    with pytest.raises(ValueError, match="not found"):
        reg.get("nonexistent")


def test_provider_registry_error_lists_available():
    from aiservices_core.providers import BaseProvider, ProviderRegistry

    class StubProvider(BaseProvider):
        def __init__(self, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            return None

    reg = ProviderRegistry()
    reg.register("alpha", StubProvider)
    with pytest.raises(ValueError, match="alpha"):
        reg.get("missing")


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
    assert len(bar.columns) == 5
    assert isinstance(bar.columns[0], SpinnerColumn)
    assert isinstance(bar.columns[1], TextColumn)
    assert isinstance(bar.columns[2], BarColumn)
    assert isinstance(bar.columns[3], TextColumn)
    assert isinstance(bar.columns[4], TimeElapsedColumn)


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


def test_setup_logging_no_exception():
    from aiservices_core.logging import setup_logging

    setup_logging(debug=False)
    setup_logging(debug=True)


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
