import os
from unittest.mock import MagicMock, patch

import pytest
from image2image.models import Image2ImageRequest


@pytest.fixture
def dummy_request(tmp_path):
    dummy_file = tmp_path / "dummy.jpg"
    dummy_file.write_bytes(b"fake image data")
    return Image2ImageRequest(
        image_path=str(dummy_file), prompt="A beautiful test image", strength=0.5
    )


def test_request_model_defaults():
    req = Image2ImageRequest(image_path="/tmp/test.jpg", prompt="test")
    assert req.strength == 0.5
    assert req.guidance_scale == 7.5
    assert req.num_inference_steps == 50
    assert req.seed is None
    assert req.negative_prompt is None


def test_response_model():
    from image2image.models import Image2ImageResponse

    resp = Image2ImageResponse(
        output_path="/tmp/out.jpg",
        metadata={"provider": "mlx"},
    )
    assert resp.output_path == "/tmp/out.jpg"
    assert resp.metadata["provider"] == "mlx"


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1", reason="Requires RUN_INTEGRATION_TESTS=1"
)
def test_mlx_provider_integration(dummy_request, tmp_path):
    from image2image.providers.mlx import MLXProvider

    provider = MLXProvider()
    out_file = tmp_path / "out.jpg"

    response = provider.generate(dummy_request, str(out_file))
    assert os.path.exists(response.output_path)
    assert response.metadata["provider"] == "mlx"
