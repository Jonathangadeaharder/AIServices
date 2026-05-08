from unittest.mock import MagicMock, patch


class TestTranscribe:
    @patch("video2subtitle.providers.registry")
    def test_calls_provider(self, mock_registry):
        from video2subtitle.client import transcribe

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.srt"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        result = transcribe("/tmp/video.mp4", "/tmp/out.srt")

        mock_registry.get.assert_called_once_with("video2subtitle.mlx")
        mock_provider.generate.assert_called_once()
        assert str(result) == "/tmp/out.srt"

    @patch("video2subtitle.providers.registry")
    def test_custom_provider(self, mock_registry):
        from video2subtitle.client import transcribe

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.srt"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        transcribe("/tmp/video.mp4", "/tmp/out.srt", provider_name="custom.provider")

        mock_registry.get.assert_called_once_with("custom.provider")

    @patch("video2subtitle.providers.registry")
    def test_passes_language_and_format(self, mock_registry):
        from video2subtitle.client import transcribe

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.output_path = "/tmp/out.vtt"
        mock_provider.generate.return_value = mock_response
        mock_registry.get.return_value = mock_provider

        transcribe("/tmp/video.mp4", "/tmp/out.vtt", language="de", output_format="vtt")

        call_args = mock_provider.generate.call_args
        req = call_args[0][0]
        assert req.language == "de"
        assert req.output_format == "vtt"
