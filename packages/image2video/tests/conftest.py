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
