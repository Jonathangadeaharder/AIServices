import os

import pytest
from text2video.models import Text2VideoRequest


@pytest.fixture
def dummy_request():
    return Text2VideoRequest(
        prompt="A test video prompt",
        width=640,
        height=640,
        num_frames=81,
        num_inference_steps=4,
        fps=16,
    )


def test_model_validation():
    """Test that model validation constraints work."""
    # Valid request
    req = Text2VideoRequest(prompt="test", width=640, height=640)
    assert req.width == 640

    # Invalid dimension (not divisible by 8)
    with pytest.raises(ValueError, match="divisible by 8"):
        Text2VideoRequest(prompt="test", width=641)

    # Invalid dimension (too small)
    with pytest.raises(ValueError, match="64-2048"):
        Text2VideoRequest(prompt="test", width=32)


def test_model_defaults():
    """Test that default values are sensible."""
    req = Text2VideoRequest(prompt="test")
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
def test_mlx_provider_integration(dummy_request, tmp_path):
    """Integration test requiring local LTX 2.3 MLX weights."""
    from text2video.providers.mlx import MLXProvider

    provider = MLXProvider()
    out_file = tmp_path / "out.mp4"

    response = provider.generate(dummy_request, str(out_file))
    assert os.path.exists(response.output_path)
    assert response.metadata["provider"] == "mlx"
