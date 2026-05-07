import sys

from image2video.models import Image2VideoRequest
from image2video.providers.mlx import MLXProvider


def test_mlx_provider_init_default():
    provider = MLXProvider.__new__(MLXProvider)
    MLXProvider.__init__(provider, model_dir="/fake/models")
    assert provider._model_dir == "/fake/models"
    assert provider._pipeline is None


def test_mlx_provider_init_custom_dir():
    provider = MLXProvider.__new__(MLXProvider)
    MLXProvider.__init__(provider, model_dir="/custom/path")
    assert provider._model_dir == "/custom/path"


def test_mlx_provider_env_var(monkeypatch):
    monkeypatch.setenv("IMAGE2VIDEO_MODEL_DIR", "/env/path")
    provider = MLXProvider.__new__(MLXProvider)
    MLXProvider.__init__(provider)
    assert provider._model_dir == "/env/path"


def test_mlx_provider_generate(mocker, tmp_path):
    mock_open = mocker.patch("PIL.Image.open")
    mock_pipeline = mocker.MagicMock()
    mock_pipeline.generate_from_image.return_value = (mocker.MagicMock(), None)

    mock_img = mocker.MagicMock()
    mock_open.return_value.__enter__ = mocker.MagicMock(return_value=mock_img)
    mock_open.return_value.__exit__ = mocker.MagicMock(return_value=False)

    provider = MLXProvider.__new__(MLXProvider)
    provider._model_dir = "/fake"
    provider._pipeline = mock_pipeline

    request = Image2VideoRequest(
        image_path="/tmp/test.png",
        prompt="test video",
        seed=42,
    )
    out = tmp_path / "out.mp4"
    response = provider.generate(request, str(out))

    assert response.output_path == str(out)
    assert response.metadata["provider"] == "mlx"
    assert response.metadata["seed"] == 42
    mock_pipeline.generate_from_image.assert_called_once()
    mock_pipeline.save_video.assert_called_once()


def test_mlx_provider_generate_default_output(mocker):
    mock_open = mocker.patch("PIL.Image.open")
    mock_pipeline = mocker.MagicMock()
    mock_pipeline.generate_from_image.return_value = (mocker.MagicMock(), None)

    mock_img = mocker.MagicMock()
    mock_open.return_value.__enter__ = mocker.MagicMock(return_value=mock_img)
    mock_open.return_value.__exit__ = mocker.MagicMock(return_value=False)

    provider = MLXProvider.__new__(MLXProvider)
    provider._model_dir = "/fake"
    provider._pipeline = mock_pipeline

    request = Image2VideoRequest(image_path="/tmp/test.png", prompt="test")
    response = provider.generate(request)

    assert response.output_path == "output.mp4"


def test_mlx_provider_generate_no_seed(mocker, tmp_path):
    mock_open = mocker.patch("PIL.Image.open")
    mock_pipeline = mocker.MagicMock()
    mock_pipeline.generate_from_image.return_value = (mocker.MagicMock(), None)

    mock_img = mocker.MagicMock()
    mock_open.return_value.__enter__ = mocker.MagicMock(return_value=mock_img)
    mock_open.return_value.__exit__ = mocker.MagicMock(return_value=False)

    provider = MLXProvider.__new__(MLXProvider)
    provider._model_dir = "/fake"
    provider._pipeline = mock_pipeline

    request = Image2VideoRequest(image_path="/tmp/test.png", prompt="test", seed=None)
    out = tmp_path / "out.mp4"
    response = provider.generate(request, str(out))

    assert isinstance(response.metadata["seed"], int)
    call_kwargs = mock_pipeline.generate_from_image.call_args[1]
    assert isinstance(call_kwargs["seed"], int)


def test_mlx_provider_load_pipeline(mocker):
    mock_module = mocker.MagicMock()
    mocker.patch.dict(
        sys.modules,
        {
            "ltx_pipelines_mlx": mock_module,
            "ltx_pipelines_mlx.ti2vid_one_stage": mock_module.ti2vid_one_stage,
        },
    )
    mock_pipeline_cls = mock_module.ti2vid_one_stage.ImageToVideoPipeline
    mock_pipeline_cls.return_value = mocker.MagicMock()

    provider = MLXProvider.__new__(MLXProvider)
    provider._model_dir = "/fake"
    provider._pipeline = None
    provider._load_pipeline()

    mock_pipeline_cls.assert_called_once_with(model_dir="/fake")
    assert provider._pipeline is not None


def test_mlx_provider_load_pipeline_cached(mocker):
    mock_module = mocker.MagicMock()
    mocker.patch.dict(
        sys.modules,
        {
            "ltx_pipelines_mlx": mock_module,
            "ltx_pipelines_mlx.ti2vid_one_stage": mock_module.ti2vid_one_stage,
        },
    )
    mock_pipeline_cls = mock_module.ti2vid_one_stage.ImageToVideoPipeline

    provider = MLXProvider.__new__(MLXProvider)
    provider._model_dir = "/fake"
    cached_pipeline = mocker.MagicMock()
    provider._pipeline = cached_pipeline

    provider._load_pipeline()
    mock_pipeline_cls.assert_not_called()
    assert provider._pipeline is cached_pipeline, "pipeline should remain cached"
