import sys

import pytest
from audio2subtitle.models import (
    Audio2SubtitleRequest,
    Audio2SubtitleResponse,
    SubtitleEntry,
)
from audio2subtitle.providers.mlx import _format_timestamp
from pydantic import ValidationError


class TestModelDefaults:
    def test_request_defaults(self):
        req = Audio2SubtitleRequest(audio_path="/tmp/audio.wav")
        assert req.audio_path == "/tmp/audio.wav"
        assert req.language is None
        assert req.output_format == "srt"
        assert req.model_name == "mlx-community/whisper-large-v3"
        assert req.word_timestamps is True

    def test_request_requires_audio_path(self):
        with pytest.raises(ValidationError):
            Audio2SubtitleRequest()

    def test_request_custom_values(self):
        req = Audio2SubtitleRequest(
            audio_path="/tmp/audio.wav",
            language="en",
            output_format="vtt",
            model_name="custom-model",
            word_timestamps=False,
        )
        assert req.language == "en"
        assert req.output_format == "vtt"
        assert req.model_name == "custom-model"
        assert req.word_timestamps is False


class TestSubtitleEntry:
    def test_subtitle_entry(self):
        entry = SubtitleEntry(index=1, start_time=0.0, end_time=2.5, text="Hello world")
        assert entry.index == 1
        assert entry.start_time == 0.0
        assert entry.end_time == 2.5
        assert entry.text == "Hello world"


class TestTimestampFormatting:
    def test_srt_timestamp(self):
        assert _format_timestamp(3661.5, "srt") == "01:01:01,500"

    def test_srt_timestamp_zero(self):
        assert _format_timestamp(0.0, "srt") == "00:00:00,000"

    def test_vtt_timestamp(self):
        assert _format_timestamp(3661.5, "vtt") == "01:01:01.500"

    def test_vtt_timestamp_zero(self):
        assert _format_timestamp(0.0, "vtt") == "00:00:00.000"

    def test_srt_timestamp_with_millis(self):
        assert _format_timestamp(65.123, "srt") == "00:01:05,123"

    def test_vtt_timestamp_with_millis(self):
        assert _format_timestamp(65.123, "vtt") == "00:01:05.123"


class TestResponse:
    def test_response_defaults(self):
        resp = Audio2SubtitleResponse(output_path="/tmp/out.srt")
        assert resp.output_path == "/tmp/out.srt"
        assert resp.entries == []
        assert resp.language is None
        assert resp.metadata == {}

    def test_response_with_entries(self):
        entries = [
            SubtitleEntry(index=1, start_time=0.0, end_time=2.5, text="Hello"),
            SubtitleEntry(index=2, start_time=2.5, end_time=5.0, text="World"),
        ]
        resp = Audio2SubtitleResponse(
            output_path="/tmp/out.srt",
            entries=entries,
            language="en",
            metadata={"provider": "mlx-whisper"},
        )
        assert len(resp.entries) == 2
        assert resp.language == "en"


def test_mlx_provider_srt(tmp_path, mocker):
    _mock_whisper = mocker.MagicMock()
    mocker.patch.dict(sys.modules, {"mlx_whisper": _mock_whisper})
    _mock_whisper.transcribe.reset_mock()
    _mock_whisper.transcribe.return_value = {
        "text": "Hello world. Goodbye.",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Hello world."},
            {"start": 2.5, "end": 5.0, "text": "Goodbye."},
        ],
    }

    from audio2subtitle.providers.mlx import MLXWhisperProvider

    provider = MLXWhisperProvider()
    request = Audio2SubtitleRequest(audio_path="/tmp/audio.wav")
    out_file = tmp_path / "out.srt"
    response = provider.generate(request, str(out_file))

    assert response.output_path == str(out_file)
    assert len(response.entries) == 2
    assert response.language == "en"
    assert response.metadata["provider"] == "mlx-whisper"
    assert out_file.exists()
    content = out_file.read_text()
    assert "1\n" in content
    assert "-->" in content
    assert "Hello world." in content
    assert "2\n" in content
    assert "Goodbye." in content

    _mock_whisper.transcribe.assert_called_once_with(
        "/tmp/audio.wav",
        path_or_hf_repo="mlx-community/whisper-large-v3",
        language=None,
        word_timestamps=True,
    )


def test_mlx_provider_vtt(tmp_path, mocker):
    _mock_whisper = mocker.MagicMock()
    mocker.patch.dict(sys.modules, {"mlx_whisper": _mock_whisper})
    _mock_whisper.transcribe.reset_mock()
    _mock_whisper.transcribe.return_value = {
        "text": "Hello.",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello."},
        ],
    }

    from audio2subtitle.providers.mlx import MLXWhisperProvider

    provider = MLXWhisperProvider()
    request = Audio2SubtitleRequest(audio_path="/tmp/audio.wav", output_format="vtt")
    out_file = tmp_path / "out.vtt"
    provider.generate(request, str(out_file))

    content = out_file.read_text()
    assert content.startswith("WEBVTT\n\n")
    assert "-->" in content
    assert "Hello." in content


def test_mlx_provider_skips_empty_segments(tmp_path, mocker):
    _mock_whisper = mocker.MagicMock()
    mocker.patch.dict(sys.modules, {"mlx_whisper": _mock_whisper})
    _mock_whisper.transcribe.reset_mock()
    _mock_whisper.transcribe.return_value = {
        "text": "Hello.",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello."},
            {"start": 1.0, "end": 2.0, "text": "   "},
            {"start": 2.0, "end": 3.0, "text": "World."},
        ],
    }

    from audio2subtitle.providers.mlx import MLXWhisperProvider

    provider = MLXWhisperProvider()
    request = Audio2SubtitleRequest(audio_path="/tmp/audio.wav")
    out_file = tmp_path / "out.srt"
    response = provider.generate(request, str(out_file))

    assert len(response.entries) == 2
    assert response.entries[0].text == "Hello."
    assert response.entries[1].text == "World."


def test_mlx_whisper_transcription_error(tmp_path, mocker):
    _mock_whisper = mocker.MagicMock()
    mocker.patch.dict(sys.modules, {"mlx_whisper": _mock_whisper})
    _mock_whisper.transcribe.reset_mock()
    _mock_whisper.transcribe.side_effect = RuntimeError("Model not found")

    from audio2subtitle.providers.mlx import MLXWhisperProvider

    provider = MLXWhisperProvider()
    request = Audio2SubtitleRequest(audio_path="/tmp/audio.wav")
    out_file = tmp_path / "out.srt"

    with pytest.raises(RuntimeError, match="Subtitle generation failed"):
        provider.generate(request, str(out_file))
