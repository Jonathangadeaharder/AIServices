import os

import pytest
from image2video.models import Image2VideoRequest


@pytest.fixture
def dummy_request():
    return Image2VideoRequest(
        image_path="/tmp/test_image.png",
        prompt="A test video prompt",
        width=640,
        height=640,
        num_frames=81,
        num_inference_steps=4,
        fps=16,
    )


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
    assert req.width == 640
    assert req.height == 640
    assert req.num_frames == 81
    assert req.num_inference_steps == 4
    assert req.fps == 16
    assert req.seed is None
    assert "static" in req.negative_prompt


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Requires RUN_INTEGRATION_TESTS=1",
)
def test_comfyui_provider_integration(dummy_request, tmp_path):
    """Integration test requiring a running ComfyUI server."""
    from image2video.providers.mlx import MLXProvider

    provider = MLXProvider()
    out_file = tmp_path / "out.mp4"

    response = provider.generate(dummy_request, str(out_file))
    assert os.path.exists(response.output_path)
