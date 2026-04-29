import pytest
from pydantic import ValidationError
from text2audio.models import AudioCategory, Text2AudioRequest, Text2AudioResponse


def test_model_defaults():
    req = Text2AudioRequest(prompt="ambient rain sounds")
    assert req.negative_prompt == ""
    assert req.duration_seconds == 10.0
    assert req.output_format == "wav"
    assert req.category == AudioCategory.MUSIC
    assert req.seed is None
    assert req.model_version == "meta/musicgen"


def test_audio_category_enum():
    assert AudioCategory.MUSIC.value == "music"
    assert AudioCategory.SFX.value == "sfx"
    assert AudioCategory.AMBIENT.value == "ambient"
    assert AudioCategory.SPEECH.value == "speech"


def test_duration_validation_too_short():
    with pytest.raises(ValidationError):
        Text2AudioRequest(prompt="test", duration_seconds=0.1)


def test_duration_validation_too_long():
    with pytest.raises(ValidationError):
        Text2AudioRequest(prompt="test", duration_seconds=301.0)


def test_duration_validation_at_bounds():
    req_min = Text2AudioRequest(prompt="test", duration_seconds=0.5)
    assert req_min.duration_seconds == 0.5

    req_max = Text2AudioRequest(prompt="test", duration_seconds=300.0)
    assert req_max.duration_seconds == 300.0


def test_response_model():
    resp = Text2AudioResponse(
        output_path="/tmp/out.wav",
        duration_seconds=10.0,
        metadata={"provider": "replicate"},
    )
    assert resp.output_path == "/tmp/out.wav"
    assert resp.duration_seconds == 10.0
    assert resp.metadata["provider"] == "replicate"


def test_replicate_generate_full(tmp_path, mocker):
    from text2audio.providers.replicate_cloud import ReplicateProvider

    provider = ReplicateProvider()
    request = Text2AudioRequest(prompt="calm piano music", duration_seconds=5.0)
    out = tmp_path / "out.wav"

    mock_response = mocker.MagicMock()
    mock_response.read.side_effect = [b"audio-data", b""]
    mock_response.__enter__ = mocker.MagicMock(return_value=mock_response)
    mock_response.__exit__ = mocker.MagicMock(return_value=False)

    mocker.patch("replicate.run", return_value=["https://example.com/audio.wav"])
    mocker.patch(
        "text2audio.providers.replicate_cloud.urllib.request.urlopen",
        return_value=mock_response,
    )

    response = provider.generate(request, str(out))

    assert response.output_path == str(out)
    assert out.read_bytes() == b"audio-data"
