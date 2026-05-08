import pytest
from pydantic import ValidationError
from text2audio.models import AudioCategory, Text2AudioRequest, Text2AudioResponse


def test_model_defaults():
    req = Text2AudioRequest(prompt="ambient rain sounds")
    assert req.category == AudioCategory.music
    assert req.duration_seconds == 10.0
    assert req.output_format == "wav"
    assert req.seed is None


def test_category_validation():
    with pytest.raises(ValidationError):
        Text2AudioRequest(prompt="test", category="invalid")


def test_output_format_validation():
    with pytest.raises(ValidationError):
        Text2AudioRequest(prompt="test", output_format="flac")


def test_duration_must_be_positive():
    with pytest.raises(ValidationError):
        Text2AudioRequest(prompt="test", duration_seconds=0)


def test_response_model():
    resp = Text2AudioResponse(
        output_path="/tmp/out.wav",
        duration_seconds=10.0,
        metadata={"provider": "test"},
    )
    assert resp.output_path == "/tmp/out.wav"
    assert resp.duration_seconds == 10.0
    assert resp.metadata["provider"] == "test"
