import os
from unittest.mock import MagicMock, patch

import pytest
from image2image.models import Image2ImageRequest
from image2image.providers.replicate_cloud import ReplicateProvider


@pytest.fixture
def dummy_request(tmp_path):
    dummy_file = tmp_path / "dummy.jpg"
    dummy_file.write_text("fake image data")
    return Image2ImageRequest(
        image_path=str(dummy_file),
        prompt="A beautiful test image",
        strength=0.5
    )


@patch("image2image.providers.replicate_cloud.replicate.run")
@patch("image2image.providers.replicate_cloud.requests.get")
@patch("image2image.providers.replicate_cloud.Image.open")
def test_replicate_provider_mocked(mock_img_open, mock_get, mock_run, dummy_request, tmp_path):
    # Setup mocks
    mock_run.return_value = ["http://example.com/out.jpg"]

    mock_response = MagicMock()
    mock_response.content = b"fake_image_data"
    mock_get.return_value = mock_response

    mock_img = MagicMock()
    mock_img_open.return_value.convert.return_value = mock_img

    provider = ReplicateProvider()

    out_file = tmp_path / "out.jpg"
    response = provider.generate(dummy_request, str(out_file))

    assert response.output_path == str(out_file)
    assert response.metadata["url"] == "http://example.com/out.jpg"
    mock_run.assert_called_once()
    mock_img.save.assert_called_once_with(str(out_file))


@pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS"), reason="Requires RUN_INTEGRATION_TESTS=1"
)
def test_local_provider_integration(dummy_request, tmp_path):
    # This test will actually load the diffusers pipeline, so it's gated.
    from image2image.providers.local_sdxl import LocalSDXLProvider

    provider = LocalSDXLProvider()
    out_file = tmp_path / "out.jpg"

    # We would need a real image_path for this to work
    dummy_request.image_path = "https://picsum.photos/512"

    response = provider.generate(dummy_request, str(out_file))
    assert os.path.exists(response.output_path)
