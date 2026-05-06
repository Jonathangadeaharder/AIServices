import pytest
from image2image.models import Image2ImageRequest


@pytest.fixture
def dummy_request(tmp_path):
    dummy_file = tmp_path / "dummy.jpg"
    dummy_file.write_bytes(b"fake image data")
    return Image2ImageRequest(
        image_path=str(dummy_file), prompt="A beautiful test image", strength=0.5
    )
