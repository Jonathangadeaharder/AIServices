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
    req = Text2VideoRequest(prompt="test", width=640, height=640)
    assert req.width == 640

    with pytest.raises(ValueError, match="divisible by 8"):
        Text2VideoRequest(prompt="test", width=641)

    with pytest.raises(ValueError, match="64-2048"):
        Text2VideoRequest(prompt="test", width=32)


def test_model_defaults():
    req = Text2VideoRequest(prompt="test")
    assert req.width == 640
    assert req.height == 640
    assert req.num_frames == 81
    assert req.num_inference_steps == 4
    assert req.fps == 16
    assert req.seed is None
    assert "static" in req.negative_prompt
