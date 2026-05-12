from image2image.models import Image2ImageRequest
from image2image.providers.mlx import MLXProvider


def test_mlx_provider_init_default():
    provider = MLXProvider()
    assert provider.model_id == MLXProvider.DEFAULT_MODEL
    assert provider._model is None


def test_mlx_provider_custom_model():
    provider = MLXProvider(model_id="custom-model")
    assert provider.model_id == "custom-model"


def test_mlx_provider_generate_with_mocked_model(mocker, tmp_path):
    mock_img = mocker.MagicMock()
    mocker.patch("image2image.providers.mlx.load_image", return_value=mock_img)

    provider = MLXProvider()
    provider._model = {"dummy": "weights"}

    request = Image2ImageRequest(image_path="/tmp/test.jpg", prompt="test", seed=42)
    out = tmp_path / "out.png"
    response = provider.generate(request, str(out))

    assert response.output_path == str(out)
    assert response.metadata["provider"] == "mlx"
    assert response.metadata["seed"] == 42
    mock_img.save.assert_called_once()


def test_mlx_provider_generate_no_seed(mocker, tmp_path):
    mock_img = mocker.MagicMock()
    mocker.patch("image2image.providers.mlx.load_image", return_value=mock_img)

    provider = MLXProvider()
    provider._model = {"dummy": "weights"}

    request = Image2ImageRequest(image_path="/tmp/test.jpg", prompt="test", seed=None)
    out = tmp_path / "out.png"
    response = provider.generate(request, str(out))

    assert isinstance(response.metadata["seed"], int)
