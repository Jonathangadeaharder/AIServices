from text2audio.models import Text2AudioRequest
from text2audio.providers.mlx import MLXProvider


def test_mlx_provider_init_default():
    provider = MLXProvider()
    assert provider is not None


def test_mlx_provider_generate_wav(tmp_path):
    provider = MLXProvider()
    request = Text2AudioRequest(
        prompt="calm piano music",
        duration_seconds=1.0,
        output_format="wav",
        seed=42,
    )
    out = tmp_path / "out.wav"
    response = provider.generate(request, str(out))

    assert response.output_path == str(out)
    assert response.metadata["provider"] == "mlx"
    assert response.metadata["seed"] == 42
    assert out.exists()
    assert out.stat().st_size > 0


def test_mlx_provider_generate_default_output():
    provider = MLXProvider()
    request = Text2AudioRequest(prompt="test", duration_seconds=1.0, seed=99)
    response = provider.generate(request, output_path=None)

    assert response.output_path == "output.wav"
    assert response.metadata["seed"] == 99


def test_mlx_provider_generate_no_seed(tmp_path):
    provider = MLXProvider()
    request = Text2AudioRequest(prompt="test", duration_seconds=1.0, seed=None)
    out = tmp_path / "out.wav"
    response = provider.generate(request, str(out))

    assert isinstance(response.metadata["seed"], int)
