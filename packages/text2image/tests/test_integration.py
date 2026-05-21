from text2image.models import Text2ImageRequest
from text2image.providers.mlx import MLXProvider


def test_mlx_provider_init_default():
    provider = MLXProvider()
    assert provider.model_name == MLXProvider.DEFAULT_MODEL
    assert provider._pipeline is None


def test_mlx_provider_generate_with_mocked_pipeline(mocker, tmp_path):
    mocker.patch("text2image.providers.mlx.shutil.which", return_value="/bin/mflux-generate")
    mock_run = mocker.patch("text2image.providers.mlx.subprocess.run")
    mock_run.return_value.returncode = 0

    provider = MLXProvider()
    request = Text2ImageRequest(prompt="a sunset", seed=42)
    out = tmp_path / "out.png"
    response = provider.generate(request, str(out))

    assert response.output_path == str(out)
    assert response.metadata["provider"] == "mlx"
    assert response.metadata["seed"] == 42
    args = mock_run.call_args.args[0]
    assert "--seed" in args
    assert "42" in args


def test_mlx_provider_generate_default_output(mocker):
    mocker.patch("text2image.providers.mlx.shutil.which", return_value="/bin/mflux-generate")
    mock_run = mocker.patch("text2image.providers.mlx.subprocess.run")
    mock_run.return_value.returncode = 0

    provider = MLXProvider()
    request = Text2ImageRequest(prompt="test")
    response = provider.generate(request, output_path=None)

    assert response.output_path == "output.png"
