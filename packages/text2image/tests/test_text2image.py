import os
from unittest.mock import MagicMock, patch

import pytest
from text2image.models import Text2ImageRequest


@pytest.fixture
def dummy_request():
    return Text2ImageRequest(
        prompt="A beautiful test image",
        guidance_scale=7.5,
        num_inference_steps=50,
        width=1024,
        height=1024,
    )


def test_request_model_defaults():
    req = Text2ImageRequest(prompt="test")
    assert req.guidance_scale == 7.5
    assert req.num_inference_steps == 50
    assert req.width == 1024
    assert req.height == 1024


def test_response_model():
    from text2image.models import Text2ImageResponse

    resp = Text2ImageResponse(
        output_path="/tmp/out.png",
        metadata={"provider": "mlx"},
    )
    assert resp.output_path == "/tmp/out.png"
    assert resp.metadata["provider"] == "mlx"


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1", reason="Requires RUN_INTEGRATION_TESTS=1"
)
def test_mlx_provider_integration(dummy_request, tmp_path):
    from text2image.providers.mlx import MLXProvider

    provider = MLXProvider()
    out_file = tmp_path / "out.png"

    response = provider.generate(dummy_request, str(out_file))
    assert os.path.exists(response.output_path)
    assert response.metadata["provider"] == "mlx"
