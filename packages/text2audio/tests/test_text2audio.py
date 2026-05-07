import pytest
from pydantic import ValidationError
from text2audio.models import Text2AudioRequest, Text2AudioResponse


def test_model_defaults():
    req = Text2AudioRequest(text="ambient rain sounds")
    assert req.voice == "default"
    assert req.speed == 1.0
    assert req.seed is None


def test_speed_validation_too_slow():
    with pytest.raises(ValidationError):
        Text2AudioRequest(text="test", speed=0.1)


def test_speed_validation_too_fast():
    with pytest.raises(ValidationError):
        Text2AudioRequest(text="test", speed=3.0)


def test_speed_validation_at_bounds():
    req_min = Text2AudioRequest(text="test", speed=0.5)
    assert req_min.speed == 0.5

    req_max = Text2AudioRequest(text="test", speed=2.0)
    assert req_max.speed == 2.0


def test_response_model():
    resp = Text2AudioResponse(
        output_path="/tmp/out.wav",
        duration_seconds=10.0,
        metadata={"provider": "test"},
    )
    assert resp.output_path == "/tmp/out.wav"
    assert resp.duration_seconds == 10.0
    assert resp.metadata["provider"] == "test"
