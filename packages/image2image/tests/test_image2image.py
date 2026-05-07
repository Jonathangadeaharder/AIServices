from image2image.models import Image2ImageRequest
from image2image.providers.mlx import MLXProvider


def test_mlx_provider_init():
    provider = MLXProvider()
    assert provider.model_id == MLXProvider.DEFAULT_MODEL
    assert provider._model is None


def test_mlx_provider_custom_model():
    provider = MLXProvider(model_id="custom-model")
    assert provider.model_id == "custom-model"


def test_mlx_provider_generate(mocker, dummy_request, tmp_path):
    mocker.patch("image2image.providers.mlx.MLXProvider._load_model", return_value=None)
    mock_img = mocker.MagicMock()
    mocker.patch("image2image.providers.mlx.load_image", return_value=mock_img)

    provider = MLXProvider()
    provider._model = {"dummy": "weights"}

    out_file = tmp_path / "out.jpg"
    response = provider.generate(dummy_request, str(out_file))

    assert response.output_path == str(out_file)
    assert response.metadata["provider"] == "mlx"
    mock_img.save.assert_called_once_with(str(out_file))


def test_request_model_defaults():
    req = Image2ImageRequest(image_path="/tmp/test.jpg", prompt="test")
    assert req.strength == 0.5, "default strength"
    assert req.guidance_scale == 7.5, "default guidance_scale"
    assert req.num_inference_steps == 50, "default num_inference_steps"
    assert req.seed is None, "default seed"
