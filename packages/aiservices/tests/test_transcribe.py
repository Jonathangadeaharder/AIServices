"""Tests for aiservices.transcribe module.

Covers:
- MLXWhisperTranscriber with mocked mlx_whisper
- FasterWhisperTranscriber with mocked faster_whisper
- create_transcriber factory with auto-detection
- Segment / TranscriptionResult dataclasses
- _fmt_time helper
- transcribe_to_srt output format
- convenience functions
"""

import pytest
from aiservices.transcribe import (
    FasterWhisperTranscriber,
    MLXWhisperTranscriber,
    Segment,
    TranscriptionResult,
    _fmt_time,
    create_transcriber,
    transcribe,
    transcribe_to_srt,
)

# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


class TestSegment:
    def test_default_values(self):
        seg = Segment()
        assert seg.start == 0.0
        assert seg.end == 0.0
        assert seg.text == ""

    def test_custom_values(self):
        seg = Segment(start=1.5, end=3.2, text="hello world")
        assert seg.start == 1.5
        assert seg.end == 3.2
        assert seg.text == "hello world"


class TestTranscriptionResult:
    def test_default_values(self):
        result = TranscriptionResult()
        assert result.text == ""
        assert result.segments == []
        assert result.language is None
        assert result.language_probability == 1.0

    def test_custom_values(self):
        segs = [Segment(start=0.0, end=1.0, text="hi")]
        result = TranscriptionResult(
            text="hi", segments=segs, language="en", language_probability=0.95
        )
        assert result.text == "hi"
        assert len(result.segments) == 1
        assert result.language == "en"
        assert result.language_probability == 0.95

    def test_segments_list_independent(self):
        """Each TranscriptionResult gets its own segments list."""
        r1 = TranscriptionResult()
        r2 = TranscriptionResult()
        r1.segments.append(Segment(text="a"))
        assert len(r2.segments) == 0


# ---------------------------------------------------------------------------
# _fmt_time helper
# ---------------------------------------------------------------------------


class TestFmtTime:
    def test_zero(self):
        assert _fmt_time(0.0) == "00:00:00,000"

    def test_one_second(self):
        assert _fmt_time(1.0) == "00:00:01,000"

    def test_complex_time(self):
        # 1h 2m 3.456s
        result = _fmt_time(3723.456)
        assert result == "01:02:03,456"

    def test_minute_boundary(self):
        assert _fmt_time(60.0) == "00:01:00,000"

    def test_hour_boundary(self):
        assert _fmt_time(3600.0) == "01:00:00,000"


# ---------------------------------------------------------------------------
# MLXWhisperTranscriber
# ---------------------------------------------------------------------------


class TestMLXWhisperTranscriber:
    def test_model_default(self):
        t = MLXWhisperTranscriber()
        assert t._model == "mlx-community/whisper-large-v3-turbo"

    def test_model_custom(self):
        t = MLXWhisperTranscriber(model="custom/whisper-small")
        assert t._model == "custom/whisper-small"

    def test_transcribe_success(self, mocker):
        mock_mlx = mocker.MagicMock()
        mock_mlx.transcribe.return_value = {
            "text": "hello world",
            "segments": [{"start": 0.0, "end": 2.0, "text": "hello world"}],
            "language": "en",
        }
        mocker.patch.dict("sys.modules", {"mlx_whisper": mock_mlx})

        t = MLXWhisperTranscriber()
        result = t.transcribe("/path/to/audio.wav", language="en")

        assert result.text == "hello world"
        assert len(result.segments) == 1
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 2.0
        assert result.language == "en"

        mock_mlx.transcribe.assert_called_once_with(
            "/path/to/audio.wav",
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
            language="en",
            word_timestamps=True,
        )

    def test_transcribe_no_language(self, mocker):
        mock_mlx = mocker.MagicMock()
        mock_mlx.transcribe.return_value = {
            "text": "transcribed text",
            "segments": [],
            "language": "de",
        }
        mocker.patch.dict("sys.modules", {"mlx_whisper": mock_mlx})

        t = MLXWhisperTranscriber()
        result = t.transcribe("/audio.wav")

        assert result.text == "transcribed text"
        assert result.language == "de"

    def test_transcribe_empty_segments(self, mocker):
        mock_mlx = mocker.MagicMock()
        mock_mlx.transcribe.return_value = {
            "text": " ",
            "segments": [],
            "language": None,
        }
        mocker.patch.dict("sys.modules", {"mlx_whisper": mock_mlx})

        t = MLXWhisperTranscriber()
        result = t.transcribe("/audio.wav")

        assert result.text == ""
        assert result.segments == []

    def test_transcribe_import_error(self, mocker):
        # Force mlx_whisper to not be importable
        import sys as _sys

        mocker.patch.dict(_sys.modules, {"mlx_whisper": None}, clear=False)

        t = MLXWhisperTranscriber()
        with pytest.raises(ImportError, match="mlx-whisper not installed"):
            t.transcribe("/audio.wav")

    def test_transcribe_batch(self, mocker):
        mock_mlx = mocker.MagicMock()
        mock_mlx.transcribe.side_effect = [
            {"text": "one", "segments": [], "language": "en"},
            {"text": "two", "segments": [], "language": "en"},
        ]
        mocker.patch.dict("sys.modules", {"mlx_whisper": mock_mlx})

        t = MLXWhisperTranscriber()
        results = t.transcribe_batch(["/a.wav", "/b.wav"], language="en")

        assert len(results) == 2
        assert results[0].text == "one"
        assert results[1].text == "two"


# ---------------------------------------------------------------------------
# FasterWhisperTranscriber
# ---------------------------------------------------------------------------


class TestFasterWhisperTranscriber:
    def test_default_init(self):
        t = FasterWhisperTranscriber()
        assert t._model_size == "tiny"
        assert t._device == "cpu"
        assert t._compute_type == "float32"
        assert t._model is None

    def test_custom_init(self):
        t = FasterWhisperTranscriber(model_size="large", device="cuda", compute_type="float16")
        assert t._model_size == "large"
        assert t._device == "cuda"
        assert t._compute_type == "float16"

    def test_device_from_env(self, monkeypatch):
        monkeypatch.setenv("WHISPER_DEVICE", "cuda")
        t = FasterWhisperTranscriber()
        assert t._device == "cuda"

    def test_get_model_lazy(self, mocker):
        mock_fw = mocker.MagicMock()
        mock_whisper_model = mocker.MagicMock()
        mock_fw.WhisperModel = mock_whisper_model

        # Patch the import at the module level
        mocker.patch.dict(
            "sys.modules",
            {"faster_whisper": mock_fw},
        )

        t = FasterWhisperTranscriber(model_size="base")
        model = t._get_model()
        model2 = t._get_model()  # Should return cached

        assert model is model2
        mock_whisper_model.assert_called_once_with("base", device="cpu", compute_type="float32")

    def test_transcribe(self, mocker):
        mock_fw = mocker.MagicMock()
        mock_model_instance = mocker.MagicMock()
        mock_fw.WhisperModel = mocker.MagicMock(return_value=mock_model_instance)
        mocker.patch.dict("sys.modules", {"faster_whisper": mock_fw})

        # Mock segments iterator
        mock_seg1 = mocker.MagicMock(start=0.0, end=1.5, text="hello")
        mock_seg2 = mocker.MagicMock(start=1.5, end=3.0, text="world")
        mock_model_instance.transcribe.return_value = (
            iter([mock_seg1, mock_seg2]),
            mocker.MagicMock(language="en", language_probability=0.99),
        )

        t = FasterWhisperTranscriber()
        result = t.transcribe("/audio.wav")

        assert result.text == "hello world"
        assert len(result.segments) == 2
        assert result.segments[0].text == "hello"
        assert result.language == "en"
        assert result.language_probability == 0.99


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateTranscriber:
    def test_explicit_mlx(self, mocker):
        mocker.patch.dict("sys.modules", {"mlx_whisper": mocker.MagicMock()})
        t = create_transcriber("mlx")
        assert isinstance(t, MLXWhisperTranscriber)

    def test_explicit_faster(self, mocker):
        mocker.patch.dict("sys.modules", {"faster_whisper": mocker.MagicMock()})
        t = create_transcriber("faster-whisper")
        assert isinstance(t, FasterWhisperTranscriber)

    def test_cpu_alias(self, mocker):
        mocker.patch.dict("sys.modules", {"faster_whisper": mocker.MagicMock()})
        t = create_transcriber("cpu")
        assert isinstance(t, FasterWhisperTranscriber)

    def test_auto_prefers_mlx(self, mocker):
        mocker.patch.dict("sys.modules", {"mlx_whisper": mocker.MagicMock()})
        t = create_transcriber("auto")
        assert isinstance(t, MLXWhisperTranscriber)

    def test_auto_falls_back_to_faster(self, mocker):
        # Make mlx_whisper import fail
        mocker.patch.dict("sys.modules", {"mlx_whisper": None}, clear=False)
        # Mock faster_whisper available
        mock_fw = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"faster_whisper": mock_fw})

        t = create_transcriber("auto")
        assert isinstance(t, FasterWhisperTranscriber)

    def test_auto_no_backend(self, mocker):
        # Both fail
        mocker.patch.dict(
            "sys.modules",
            {"mlx_whisper": None, "faster_whisper": None},
            clear=False,
        )
        with pytest.raises(ImportError, match="No whisper backend"):
            create_transcriber("auto")

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown transcriber"):
            create_transcriber("nonexistent")

    def test_from_env_var(self, monkeypatch, mocker):
        monkeypatch.setenv("WHISPER_BACKEND", "mlx")
        mocker.patch.dict("sys.modules", {"mlx_whisper": mocker.MagicMock()})
        t = create_transcriber()
        assert isinstance(t, MLXWhisperTranscriber)


# ---------------------------------------------------------------------------
# transcribe_to_srt
# ---------------------------------------------------------------------------


class TestTranscribeToSrt:
    def test_srt_output_format(self, tmp_path, mocker):
        mock_mlx = mocker.MagicMock()
        mock_mlx.transcribe.return_value = {
            "text": "hello world",
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "hello"},
                {"start": 2.5, "end": 5.0, "text": "world"},
            ],
            "language": "en",
        }
        mocker.patch.dict("sys.modules", {"mlx_whisper": mock_mlx})

        out_file = tmp_path / "subtitles.srt"
        result_path = transcribe_to_srt("/audio.wav", out_file, language="en", provider="mlx")

        assert result_path == out_file
        assert out_file.exists()

        content = out_file.read_text()
        lines = content.split("\n")
        assert lines[0] == "1"
        assert "00:00:00,000 --> 00:00:02,500" in lines[1]
        assert lines[2] == "hello"
        assert lines[4] == "2"
        assert lines[6] == "world"

    def test_srt_creates_parent_dirs(self, tmp_path, mocker):
        mock_mlx = mocker.MagicMock()
        mock_mlx.transcribe.return_value = {
            "text": "test",
            "segments": [{"start": 0.0, "end": 1.0, "text": "test"}],
            "language": "en",
        }
        mocker.patch.dict("sys.modules", {"mlx_whisper": mock_mlx})

        out_file = tmp_path / "nested" / "deep" / "out.srt"
        transcribe_to_srt("/audio.wav", out_file, provider="mlx")
        assert out_file.exists()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestTranscribeConvenience:
    def test_transcribe_function(self, mocker):
        mock_mlx = mocker.MagicMock()
        mock_mlx.transcribe.return_value = {
            "text": "convenience test",
            "segments": [],
            "language": "en",
        }
        mocker.patch.dict("sys.modules", {"mlx_whisper": mock_mlx})

        result = transcribe("/audio.wav", language="en", provider="mlx")
        assert result.text == "convenience test"
