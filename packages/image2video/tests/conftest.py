import pytest
from image2video.models import Image2VideoRequest


@pytest.fixture
def dummy_request(tmp_path):
    dummy_file = tmp_path / "test_image.png"
    dummy_file.write_bytes(b"fake image data")
    return Image2VideoRequest(
        image_path=str(dummy_file),
        prompt="A test video prompt",
        width=640,
        height=640,
        num_frames=81,
        num_inference_steps=4,
        fps=16,
    )
