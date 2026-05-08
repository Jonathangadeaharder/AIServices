import numpy as np
from text2image.models import Text2ImageRequest
from text2image.providers.mlx import MLXProvider


def test_mlx_provider_init_default():
    provider = MLXProvider()
    assert provider.model_name == MLXProvider.DEFAULT_MODEL
    assert provider._pipeline is None


def test_mlx_provider_custom_model():
    provider = MLXProvider(model_name="flux-dev")
    assert provider.model_name == "flux-dev"


def test_mlx_provider_generate_with_mocked_pipeline(mocker, tmp_path):
    mock_fromarray = mocker.patch("text2image.providers.mlx.Image.fromarray")
    mock_pipeline = mocker.MagicMock()
    mock_pipeline.generate_images.return_value = [np.random.rand(1024, 1024, 3).astype(np.float32)]
    mock_img = mocker.MagicMock()
    mock_fromarray.return_value = mock_img

    provider = MLXProvider()
    provider._pipeline = mock_pipeline

    request = Text2ImageRequest(prompt="a sunset", seed=42)
    out = tmp_path / "out.png"
    response = provider.generate(request, str(out))

    assert response.output_path == str(out)
    assert response.metadata["provider"] == "mlx"
    assert response.metadata["seed"] == 42
    mock_pipeline.generate_images.assert_called_once()
    mock_img.save.assert_called_once()


def test_mlx_provider_generate_default_output(mocker):
    mock_fromarray = mocker.patch("text2image.providers.mlx.Image.fromarray")
    mock_pipeline = mocker.MagicMock()
    mock_pipeline.generate_images.return_value = [np.random.rand(1024, 1024, 3).astype(np.float32)]
    mock_img = mocker.MagicMock()
    mock_fromarray.return_value = mock_img

    provider = MLXProvider()
    provider._pipeline = mock_pipeline

    request = Text2ImageRequest(prompt="test")
    response = provider.generate(request, output_path=None)

    assert response.output_path == "output.png"
