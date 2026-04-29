from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from text2image.models import Text2ImageRequest
from text2image.providers.mlx import MLXProvider


@pytest.fixture
def dummy_request():
    return Text2ImageRequest(
        prompt="A beautiful test image",
        guidance_scale=7.5,
        num_inference_steps=4,
        width=1024,
        height=1024,
    )


def test_mlx_provider_init():
    provider = MLXProvider()
    assert provider.model_name == MLXProvider.DEFAULT_MODEL
    assert provider._pipeline is None


def test_mlx_provider_custom_model():
    provider = MLXProvider(model_name="flux-dev")
    assert provider.model_name == "flux-dev"


@patch("text2image.providers.mlx.Image.fromarray")
@patch("text2image.providers.mlx.MLXProvider._load_pipeline")
def test_mlx_provider_generate(mock_load, mock_fromarray, dummy_request, tmp_path):
    mock_load.return_value = None

    mock_pipeline = MagicMock()
    mock_pipeline.generate_images.return_value = [np.random.rand(1024, 1024, 3).astype(np.float32)]
    mock_img = MagicMock()
    mock_fromarray.return_value = mock_img

    provider = MLXProvider()
    provider._pipeline = mock_pipeline

    out_file = tmp_path / "out.png"
    response = provider.generate(dummy_request, str(out_file))

    assert response.output_path == str(out_file)
    assert response.metadata["provider"] == "mlx"
    mock_pipeline.generate_images.assert_called_once()
    mock_img.save.assert_called_once()


def test_request_model_defaults():
    req = Text2ImageRequest(prompt="test")
    assert req.width == 1024
    assert req.height == 1024
    assert req.guidance_scale == 7.5
    assert req.num_inference_steps == 50
    assert req.seed is None
