import pytest
from aiservices_core.errors import ProviderError
from text2audio.models import Text2AudioRequest
from text2audio.providers.replicate_cloud import ReplicateProvider


def test_extract_url_from_list():
    provider = ReplicateProvider()
    url = provider._extract_url(["https://example.com/audio.wav"])
    assert url == "https://example.com/audio.wav"


def test_extract_url_from_dict_url():
    provider = ReplicateProvider()
    url = provider._extract_url({"url": "https://example.com/out.wav"})
    assert url == "https://example.com/out.wav"


def test_extract_url_from_dict_audio():
    provider = ReplicateProvider()
    url = provider._extract_url({"audio": "https://example.com/audio.wav"})
    assert url == "https://example.com/audio.wav"


def test_extract_url_from_dict_output():
    provider = ReplicateProvider()
    url = provider._extract_url({"output": "https://example.com/out.wav"})
    assert url == "https://example.com/out.wav"


def test_extract_url_from_dict_list():
    provider = ReplicateProvider()
    url = provider._extract_url({"url": ["https://example.com/audio.wav"]})
    assert url == "https://example.com/audio.wav"


def test_extract_url_from_string():
    provider = ReplicateProvider()
    url = provider._extract_url("https://example.com/direct.wav")
    assert url == "https://example.com/direct.wav"


def test_extract_url_empty_raises():
    provider = ReplicateProvider()
    with pytest.raises(ProviderError, match="Failed to parse"):
        provider._extract_url([])


def test_extract_url_none_raises():
    provider = ReplicateProvider()
    with pytest.raises(ProviderError, match="Failed to parse"):
        provider._extract_url({})


def test_generate_success(tmp_path, mocker):
    mock_run = mocker.patch("replicate.run")
    mock_urlopen = mocker.patch("text2audio.providers.replicate_cloud.urllib.request.urlopen")
    mock_run.return_value = ["https://example.com/audio.wav"]

    mock_response = mocker.MagicMock()
    mock_response.read.side_effect = [b"audio-data", b""]
    mock_response.__enter__ = mocker.MagicMock(return_value=mock_response)
    mock_response.__exit__ = mocker.MagicMock(return_value=False)
    mock_urlopen.return_value = mock_response

    provider = ReplicateProvider()
    request = Text2AudioRequest(prompt="calm piano", duration_seconds=5.0)
    out = tmp_path / "out.wav"
    response = provider.generate(request, str(out))

    assert response.output_path == str(out)
    assert response.metadata["provider"] == "replicate"


def test_generate_replicate_error(tmp_path, mocker):
    mock_run = mocker.patch("replicate.run")
    mock_run.side_effect = RuntimeError("API error")

    provider = ReplicateProvider()
    request = Text2AudioRequest(prompt="test")
    with pytest.raises(ProviderError, match="Replicate audio generation failed"):
        provider.generate(request, str(tmp_path / "out.wav"))


def test_generate_download_error(tmp_path, mocker):
    mock_run = mocker.patch("replicate.run")
    mock_urlopen = mocker.patch("text2audio.providers.replicate_cloud.urllib.request.urlopen")
    mock_run.return_value = ["https://example.com/audio.wav"]
    mock_urlopen.side_effect = RuntimeError("download failed")

    provider = ReplicateProvider()
    request = Text2AudioRequest(prompt="test")
    with pytest.raises(ProviderError, match="Failed to download"):
        provider.generate(request, str(tmp_path / "out.wav"))


def test_init_warns_without_token(mocker):
    mocker.patch.dict("os.environ", {}, clear=True)
    provider = ReplicateProvider()
    assert provider.device == "auto"
