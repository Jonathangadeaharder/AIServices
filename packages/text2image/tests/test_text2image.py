import pytest
from text2image.models import Text2ImageRequest
from text2image.providers.mlx import MLXProvider


def test_mlx_provider_init():
    provider = MLXProvider()
    assert provider.model_name == MLXProvider.DEFAULT_MODEL
    assert provider._pipeline is None


def test_mlx_provider_custom_model():
    provider = MLXProvider(model_name="dev")
    assert provider.model_name == "dev"


def test_mlx_provider_generate(dummy_request, tmp_path, mocker):
    mocker.patch("text2image.providers.mlx.shutil.which", return_value="/bin/mflux-generate")
    mock_run = mocker.patch("text2image.providers.mlx.subprocess.run")
    mock_run.return_value.returncode = 0

    provider = MLXProvider()
    out_file = tmp_path / "out.png"
    response = provider.generate(dummy_request, str(out_file))

    assert response.output_path == str(out_file)
    assert response.metadata["provider"] == "mlx"
    assert response.metadata["model_name"] == "schnell"
    args = mock_run.call_args.args[0]
    assert args[:3] == ["/bin/mflux-generate", "--model", "schnell"]
    assert "--prompt" in args
    assert "--width" in args
    assert "1024" in args


def test_mlx_provider_generate_default_output(dummy_request, mocker):
    mocker.patch("text2image.providers.mlx.shutil.which", return_value="/bin/mflux-generate")
    mock_run = mocker.patch("text2image.providers.mlx.subprocess.run")
    mock_run.return_value.returncode = 0

    provider = MLXProvider()
    response = provider.generate(dummy_request)

    assert response.output_path == "output.png"


def test_mlx_provider_load_pipeline(mocker):
    mocker.patch("text2image.providers.mlx.shutil.which", return_value="/bin/mflux-generate")

    provider = MLXProvider()
    provider._load_pipeline()

    assert provider._pipeline == "/bin/mflux-generate"


def test_mlx_provider_load_pipeline_missing_executable(mocker):
    mocker.patch("text2image.providers.mlx.shutil.which", return_value=None)

    provider = MLXProvider()

    with pytest.raises(ImportError, match="mflux-generate is not installed"):
        provider._load_pipeline()


def test_mlx_provider_load_pipeline_cached(mocker):
    mock_which = mocker.patch("text2image.providers.mlx.shutil.which")

    provider = MLXProvider()
    provider._pipeline = "/bin/mflux-generate"
    provider._load_pipeline()

    mock_which.assert_not_called()
    assert provider._pipeline == "/bin/mflux-generate"


def test_request_model_defaults():
    req = Text2ImageRequest(prompt="test")
    assert req.width == 1024, "default width"
    assert req.height == 1024, "default height"
    assert req.guidance_scale == 7.5, "default guidance_scale"
    assert req.num_inference_steps == 50, "default num_inference_steps"
    assert req.seed is None, "default seed"


def test_request_model_dimension_validation():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="must be >= 512") as exc_info:
        Text2ImageRequest(prompt="test", width=256)
    assert "width" in str(exc_info.value)

    with pytest.raises(ValidationError, match="must be divisible by 8") as exc_info:
        Text2ImageRequest(prompt="test", width=513)
    assert "width" in str(exc_info.value)

    with pytest.raises(ValidationError, match="must be >= 512") as exc_info:
        Text2ImageRequest(prompt="test", height=100)
    assert "height" in str(exc_info.value)

    with pytest.raises(ValidationError, match="must be divisible by 8") as exc_info:
        Text2ImageRequest(prompt="test", height=513)
    assert "height" in str(exc_info.value)
