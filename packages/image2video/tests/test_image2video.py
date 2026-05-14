import pytest
from image2video.models import Image2VideoRequest


def test_model_validation():
    """Test that model validation constraints work."""
    req = Image2VideoRequest(image_path="/tmp/test.png", prompt="test", width=640, height=640)
    assert req.width == 640

    with pytest.raises(ValueError, match="divisible by 8"):
        Image2VideoRequest(image_path="/tmp/test.png", prompt="test", width=641)

    with pytest.raises(ValueError, match="64-2048"):
        Image2VideoRequest(image_path="/tmp/test.png", prompt="test", width=32)


def test_model_defaults():
    """Test that default values are sensible."""
    req = Image2VideoRequest(image_path="/tmp/test.png", prompt="test")
    assert req.width == 640, "default width"
    assert req.height == 640, "default height"
    assert req.num_frames == 81, "default num_frames"
    assert req.num_inference_steps == 4, "default num_inference_steps"
    assert req.fps == 24, "default fps"
    assert req.seed is None, "default seed"
    assert "static" in req.negative_prompt, "default negative_prompt contains static"


def test_mlx_provider_generate_full_flow(dummy_request, tmp_path, mocker):
    from image2video.providers.mlx import MLXProvider

    mock_pipeline = mocker.MagicMock()
    provider = MLXProvider.__new__(MLXProvider)
    provider._model_dir = "/fake"
    provider._pipeline = mock_pipeline

    out_file = tmp_path / "out.mp4"

    mock_open = mocker.patch("PIL.Image.open")
    mock_img = mocker.MagicMock()
    mock_open.return_value.__enter__ = mocker.MagicMock(return_value=mock_img)
    mock_open.return_value.__exit__ = mocker.MagicMock(return_value=False)
    response = provider.generate(dummy_request, str(out_file))

    assert response.output_path == str(out_file)
    mock_pipeline.generate_and_save.assert_called_once()
