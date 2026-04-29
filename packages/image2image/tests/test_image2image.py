from unittest.mock import MagicMock, patch

import pytest
from image2image.models import Image2ImageRequest
from image2image.providers.mlx import MLXProvider


@pytest.fixture
def dummy_request(tmp_path):
    dummy_file = tmp_path / "dummy.jpg"
    dummy_file.write_bytes(b"fake image data")
    return Image2ImageRequest(
        image_path=str(dummy_file), prompt="A beautiful test image", strength=0.5
    )


def test_mlx_provider_init():
    provider = MLXProvider()
    assert provider.model_id == MLXProvider.DEFAULT_MODEL
    assert provider._model is None


def test_mlx_provider_custom_model():
    provider = MLXProvider(model_id="custom-model")
    assert provider.model_id == "custom-model"


@patch("image2image.providers.mlx.Image.open")
@patch("image2image.providers.mlx.MLXProvider._load_model")
def test_mlx_provider_generate(mock_load, mock_img_open, dummy_request, tmp_path):
    mock_load.return_value = None
    mock_img = MagicMock()
    mock_img_open.return_value.convert.return_value = mock_img

    provider = MLXProvider()
    provider._model = {"dummy": "weights"}

    out_file = tmp_path / "out.jpg"
    response = provider.generate(dummy_request, str(out_file))

    assert response.output_path == str(out_file)
    assert response.metadata["provider"] == "mlx"
    mock_img.save.assert_called_once_with(str(out_file))


def test_request_model_defaults():
    req = Image2ImageRequest(image_path="/tmp/test.jpg", prompt="test")
    assert req.strength == 0.5
    assert req.guidance_scale == 7.5
    assert req.num_inference_steps == 50
    assert req.seed is None
