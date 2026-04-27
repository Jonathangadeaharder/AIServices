import copy
import os
from unittest.mock import MagicMock, patch

import pytest
from text2image.models import Text2ImageRequest
from text2image.providers.replicate_cloud import ReplicateProvider


@pytest.fixture
def dummy_request():
    return Text2ImageRequest(
        prompt="A beautiful test image",
        guidance_scale=7.5,
        num_inference_steps=50,
        width=1024,
        height=1024,
    )


@patch("text2image.providers.replicate_cloud.replicate.run")
@patch("text2image.providers.replicate_cloud.requests.get")
@patch("text2image.providers.replicate_cloud.Image.open")
def test_replicate_provider_mocked(mock_img_open, mock_get, mock_run, dummy_request, tmp_path):
    # Setup mocks
    mock_run.return_value = ["https://example.com/out.png"]

    mock_response = MagicMock()
    mock_response.content = b"fake_image_data"
    mock_get.return_value = mock_response

    mock_img = MagicMock()
    mock_img_open.return_value.convert.return_value = mock_img

    provider = ReplicateProvider()

    out_file = tmp_path / "out.png"
    response = provider.generate(dummy_request, str(out_file))

    assert response.output_path == str(out_file)
    assert response.metadata["url"] == "https://example.com/out.png"
    mock_run.assert_called_once()
    mock_img.save.assert_called_once_with(str(out_file))


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1", reason="Requires RUN_INTEGRATION_TESTS=1"
)
def test_local_provider_integration(dummy_request, tmp_path):
    # This test will actually execute via ComfyUI WebSocket, so it's gated.
    from text2image.providers.comfyui import ComfyUIProvider

    provider = ComfyUIProvider()
    out_file = tmp_path / "out.png"

    new_request = copy.copy(dummy_request)
    new_request.num_inference_steps = 1  # fast inference for test

    response = provider.generate(new_request, str(out_file))
    assert os.path.exists(response.output_path)
