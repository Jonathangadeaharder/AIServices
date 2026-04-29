import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from video2subtitle.models import (
    SubtitleEntry,
    Video2SubtitleRequest,
    Video2SubtitleResponse,
)

_mock_whisper = MagicMock()


class TestModelDefaults:
    def test_request_defaults(self):
        req = Video2SubtitleRequest(video_path="/tmp/video.mp4")
        assert req.video_path == "/tmp/video.mp4"
        assert req.language is None
        assert req.output_format == "srt"
        assert req.model_name == "mlx-community/whisper-large-v3-turbo"
        assert req.word_timestamps is True

    def test_request_custom_values(self):
        req = Video2SubtitleRequest(
            video_path="/tmp/video.mp4",
            language="en",
            output_format="vtt",
            model_name="mlx-community/whisper-small",
            word_timestamps=False,
        )
        assert req.language == "en"
        assert req.output_format == "vtt"
        assert req.model_name == "mlx-community/whisper-small"
        assert req.word_timestamps is False

    def test_request_requires_video_path(self):
        with pytest.raises(ValidationError):
            Video2SubtitleRequest()


class TestSubtitleEntry:
    def test_subtitle_entry(self):
        entry = SubtitleEntry(index=1, start_time=0.0, end_time=2.5, text="Hello")
        assert entry.index == 1
        assert entry.start_time == 0.0
        assert entry.end_time == 2.5
        assert entry.text == "Hello"

    def test_response_with_entries(self):
        entries = [
            SubtitleEntry(index=1, start_time=0.0, end_time=2.5, text="Hello"),
            SubtitleEntry(index=2, start_time=2.5, end_time=5.0, text="World"),
        ]
        resp = Video2SubtitleResponse(
            output_path="/tmp/out.srt",
            entries=entries,
            language="en",
        )
        assert len(resp.entries) == 2
        assert resp.entries[0].text == "Hello"
        assert resp.language == "en"

    def test_response_defaults(self):
        resp = Video2SubtitleResponse(output_path="/tmp/out.srt")
        assert resp.entries == []
        assert resp.language is None
        assert resp.metadata == {}


class TestTimestampFormatting:
    def test_srt_timestamp(self):
        from video2subtitle.providers.mlx import MLXProvider

        assert MLXProvider._format_timestamp(3661.5, "srt") == "01:01:01,500"

    def test_vtt_timestamp(self):
        from video2subtitle.providers.mlx import MLXProvider

        assert MLXProvider._format_timestamp(3661.5, "vtt") == "01:01:01.500"

    def test_zero_timestamp(self):
        from video2subtitle.providers.mlx import MLXProvider

        assert MLXProvider._format_timestamp(0.0, "srt") == "00:00:00,000"

    def test_sub_second_timestamp(self):
        from video2subtitle.providers.mlx import MLXProvider

        assert MLXProvider._format_timestamp(0.123, "srt") == "00:00:00,123"


@patch("video2subtitle.providers.mlx.subprocess.run")
@patch.dict(sys.modules, {"mlx_whisper": _mock_whisper})
@patch("video2subtitle.providers.mlx.shutil.which", return_value="/usr/bin/ffmpeg")
def test_mlx_provider_srt_output(mock_which, mock_run, tmp_path):
    _mock_whisper.transcribe.reset_mock()
    _mock_whisper.transcribe.return_value = {
        "text": "Hello world",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
        ],
    }

    from video2subtitle.providers.mlx import MLXProvider

    provider = MLXProvider()
    request = Video2SubtitleRequest(video_path="/tmp/video.mp4")
    output = str(tmp_path / "out.srt")

    response = provider.generate(request, output)

    assert response.output_path == output
    assert len(response.entries) == 1
    assert response.entries[0].text == "Hello world"
    assert response.language == "en"

    content = (tmp_path / "out.srt").read_text()
    assert "00:00:00,000 --> 00:00:02,500" in content
    assert "Hello world" in content


@patch("video2subtitle.providers.mlx.subprocess.run")
@patch.dict(sys.modules, {"mlx_whisper": _mock_whisper})
@patch("video2subtitle.providers.mlx.shutil.which", return_value="/usr/bin/ffmpeg")
def test_mlx_provider_vtt_output(mock_which, mock_run, tmp_path):
    _mock_whisper.transcribe.reset_mock()
    _mock_whisper.transcribe.return_value = {
        "text": "Test",
        "language": "en",
        "segments": [
            {"start": 1.0, "end": 3.0, "text": "Test"},
        ],
    }

    from video2subtitle.providers.mlx import MLXProvider

    provider = MLXProvider()
    request = Video2SubtitleRequest(video_path="/tmp/video.mp4", output_format="vtt")
    output = str(tmp_path / "out.vtt")

    provider.generate(request, output)

    content = (tmp_path / "out.vtt").read_text()
    assert content.startswith("WEBVTT")
    assert "00:00:01.000 --> 00:00:03.000" in content


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Requires RUN_INTEGRATION_TESTS=1",
)
def test_mlx_whisper_integration(tmp_path):
    video_path = os.environ.get("VIDEO2SUBTITLE_TEST_VIDEO")
    if not video_path:
        pytest.skip("Set VIDEO2SUBTITLE_TEST_VIDEO to a real video file path")

    from video2subtitle.providers.mlx import MLXProvider

    request = Video2SubtitleRequest(video_path=video_path)
    provider = MLXProvider()
    out_file = str(tmp_path / "out.srt")

    response = provider.generate(request, out_file)
    assert len(response.entries) > 0
    assert (tmp_path / "out.srt").exists()
