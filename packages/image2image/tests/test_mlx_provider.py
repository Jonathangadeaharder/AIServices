from image2image.models import Image2ImageRequest
from image2image.providers.mlx import MLXProvider


def test_mlx_provider_load_model(mocker):
    mock_f = mocker.MagicMock()
    mock_f.keys.return_value = ["tensor1", "tensor2"]
    mock_f.get_tensor.side_effect = lambda k: mocker.MagicMock()

    mock_safe_open = mocker.MagicMock(return_value=mock_f)
    mock_f.__enter__ = mocker.MagicMock(return_value=mock_f)
    mock_f.__exit__ = mocker.MagicMock(return_value=False)

    mocker.patch("huggingface_hub.hf_hub_download", return_value="/fake/path")
    mocker.patch("safetensors.safe_open", mock_safe_open)
    provider = MLXProvider()
    provider._load_model()
    assert provider._model is not None
    assert len(provider._model) == 2


def test_mlx_provider_load_model_cached():
    provider = MLXProvider()
    provider._model = {"already": "loaded"}
    provider._load_model()
    assert provider._model == {"already": "loaded"}


def test_mlx_provider_generate_file_not_found(tmp_path):
    provider = MLXProvider()
    provider._model = {"dummy": True}

    request = Image2ImageRequest(
        image_path="/nonexistent/file.png",
        prompt="test",
    )
    out = tmp_path / "out.png"
    import pytest

    with pytest.raises(FileNotFoundError, match="Input image not found"):
        provider.generate(request, str(out))


def test_mlx_provider_generate_with_seed(mocker, tmp_path):
    mock_open = mocker.patch("image2image.providers.mlx.Image.open")
    mock_img = mocker.MagicMock()
    mock_open.return_value.convert.return_value = mock_img

    provider = MLXProvider()
    provider._model = {"dummy": True}

    dummy_input = tmp_path / "input.png"
    dummy_input.write_bytes(b"fake")

    request = Image2ImageRequest(
        image_path=str(dummy_input),
        prompt="test",
        seed=12345,
    )
    out = tmp_path / "out.png"
    response = provider.generate(request, str(out))

    assert response.metadata["seed"] == 12345
    mock_img.save.assert_called_once()
