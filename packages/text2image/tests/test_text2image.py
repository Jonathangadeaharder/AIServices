import sys

import numpy as np
import pytest
from text2image.models import Text2ImageRequest
from text2image.providers.mlx import MLXProvider


def test_mlx_provider_init():
    provider = MLXProvider()
    assert provider.model_name == MLXProvider.DEFAULT_MODEL
    assert provider._pipeline is None


def test_mlx_provider_custom_model():
    provider = MLXProvider(model_name="flux-dev")
    assert provider.model_name == "flux-dev"


def test_mlx_provider_generate(dummy_request, tmp_path, mocker):
    mocker.patch("text2image.providers.mlx.MLXProvider._load_pipeline", return_value=None)
    mock_fromarray = mocker.patch("text2image.providers.mlx.Image.fromarray")

    mock_pipeline = mocker.MagicMock()
    mock_pipeline.generate_images.return_value = [np.random.rand(1024, 1024, 3).astype(np.float32)]
    mock_img = mocker.MagicMock()
    mock_fromarray.return_value = mock_img

    provider = MLXProvider()
    provider._pipeline = mock_pipeline

    out_file = tmp_path / "out.png"
    response = provider.generate(dummy_request, str(out_file))

    assert response.output_path == str(out_file)
    assert response.metadata["provider"] == "mlx"
    mock_pipeline.generate_images.assert_called_once()
    mock_img.save.assert_called_once()


def test_mlx_provider_generate_default_output(dummy_request, mocker):
    mocker.patch("text2image.providers.mlx.MLXProvider._load_pipeline", return_value=None)
    mock_fromarray = mocker.patch("text2image.providers.mlx.Image.fromarray")

    mock_pipeline = mocker.MagicMock()
    mock_pipeline.generate_images.return_value = [np.random.rand(1024, 1024, 3).astype(np.float32)]
    mock_img = mocker.MagicMock()
    mock_fromarray.return_value = mock_img

    provider = MLXProvider()
    provider._pipeline = mock_pipeline

    response = provider.generate(dummy_request)

    assert response.output_path == "output.png"


def test_mlx_provider_load_pipeline(mocker):
    mock_flux = mocker.MagicMock()
    mocker.patch.dict(
        sys.modules,
        {"image2image": mocker.MagicMock(), "image2image.flux_mlx": mock_flux},
    )
    mock_pipeline_cls = mock_flux.FluxPipeline
    mock_pipeline_cls.return_value = mocker.MagicMock()

    provider = MLXProvider()
    provider._pipeline = None
    provider._load_pipeline()

    mock_pipeline_cls.assert_called_once_with(provider.model_name)
    assert provider._pipeline is not None


def test_mlx_provider_load_pipeline_cached(mocker):
    mock_flux = mocker.MagicMock()
    mocker.patch.dict(
        sys.modules,
        {"image2image": mocker.MagicMock(), "image2image.flux_mlx": mock_flux},
    )
    mock_pipeline_cls = mock_flux.FluxPipeline

    provider = MLXProvider()
    provider._pipeline = mocker.MagicMock()

    provider._load_pipeline()
    mock_pipeline_cls.assert_not_called()
    assert provider._pipeline is not None, "pipeline should remain cached"


def test_request_model_defaults():
    req = Text2ImageRequest(prompt="test")
    assert req.width == 1024, "default width"
    assert req.height == 1024, "default height"
    assert req.guidance_scale == 7.5, "default guidance_scale"
    assert req.num_inference_steps == 50, "default num_inference_steps"
    assert req.seed is None, "default seed"


def test_request_model_dimension_validation():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="must be >= 512") as exc_info:
        Text2ImageRequest(prompt="test", width=256)
    assert "width" in str(exc_info.value)

    with pytest.raises(ValidationError, match="must be divisible by 8") as exc_info:
        Text2ImageRequest(prompt="test", width=513)
    assert "width" in str(exc_info.value)

    with pytest.raises(ValidationError, match="must be >= 512") as exc_info:
        Text2ImageRequest(prompt="test", height=100)
    assert "height" in str(exc_info.value)

    with pytest.raises(ValidationError, match="must be divisible by 8") as exc_info:
        Text2ImageRequest(prompt="test", height=513)
    assert "height" in str(exc_info.value)
