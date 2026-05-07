import pytest
from text2image.models import Text2ImageRequest


@pytest.fixture
def dummy_request():
    return Text2ImageRequest(
        prompt="A beautiful test image",
        guidance_scale=7.5,
        num_inference_steps=4,
        width=1024,
        height=1024,
    )
