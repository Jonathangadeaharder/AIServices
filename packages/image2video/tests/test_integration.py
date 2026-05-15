from image2video.models import Image2VideoRequest
from image2video.providers.mlx import MLXProvider


def test_mlx_provider_init_default():
    provider = MLXProvider(model_dir="/fake/models")
    assert provider._model_dir == "/fake/models"
    assert provider._pipeline is None


def test_mlx_provider_env_var(monkeypatch):
    monkeypatch.setenv("IMAGE2VIDEO_MODEL_DIR", "/env/path")
    provider = MLXProvider()
    assert provider._model_dir == "/env/path"


def test_mlx_provider_generate_with_mocked_pipeline(mocker, tmp_path):
    mock_pipeline = mocker.MagicMock()
    mock_img = mocker.MagicMock()
    mock_open = mocker.patch("PIL.Image.open")
    mock_open.return_value.__enter__ = mocker.MagicMock(return_value=mock_img)
    mock_open.return_value.__exit__ = mocker.MagicMock(return_value=False)

    provider = MLXProvider(model_dir="/fake")
    provider._pipeline = mock_pipeline

    request = Image2VideoRequest(
        image_path="/tmp/test.png",
        prompt="a test video",
        seed=42,
    )
    out = tmp_path / "out.mp4"
    response = provider.generate(request, str(out))

    assert response.output_path == str(out)
    assert response.metadata["provider"] == "mlx"
    assert response.metadata["seed"] == 42
    mock_pipeline.generate_and_save.assert_called_once()


def test_mlx_provider_generate_default_output(mocker):
    mock_pipeline = mocker.MagicMock()
    mock_img = mocker.MagicMock()
    mock_open = mocker.patch("PIL.Image.open")
    mock_open.return_value.__enter__ = mocker.MagicMock(return_value=mock_img)
    mock_open.return_value.__exit__ = mocker.MagicMock(return_value=False)

    provider = MLXProvider(model_dir="/fake")
    provider._pipeline = mock_pipeline

    request = Image2VideoRequest(image_path="/tmp/test.png", prompt="test")
    response = provider.generate(request, output_path=None)

    assert response.output_path == "output.mp4"
