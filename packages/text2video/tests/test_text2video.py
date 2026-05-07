import pytest
from text2video.models import Text2VideoRequest


@pytest.fixture
def dummy_request():
    return Text2VideoRequest(
        prompt="A test video prompt",
        width=704,
        height=480,
        num_frames=97,
        num_inference_steps=8,
        fps=24,
    )


def test_model_validation():
    req = Text2VideoRequest(prompt="test", width=704, height=480)
    assert req.width == 704

    with pytest.raises(ValueError, match="divisible by 8"):
        Text2VideoRequest(prompt="test", width=641)

    with pytest.raises(ValueError, match="64-2048"):
        Text2VideoRequest(prompt="test", width=32)


def test_model_defaults():
    req = Text2VideoRequest(prompt="test")
    assert req.width == 704
    assert req.height == 480
    assert req.num_frames == 97
    assert req.num_inference_steps == 8
    assert req.fps == 24
    assert req.seed is None
    assert "static" in req.negative_prompt


def test_model_frames_validation():
    Text2VideoRequest(prompt="test", num_frames=97)
    Text2VideoRequest(prompt="test", num_frames=9)

    with pytest.raises(ValueError, match="8k\\+1"):
        Text2VideoRequest(prompt="test", num_frames=10)

    with pytest.raises(ValueError, match="8k\\+1"):
        Text2VideoRequest(prompt="test", num_frames=100)


def test_mlx_provider_generate_full_flow(dummy_request, tmp_path, mocker):
    from text2video.providers.mlx import MLXProvider

    mock_pipeline = mocker.MagicMock()

    provider = MLXProvider.__new__(MLXProvider)
    provider._model_dir = "/fake"
    provider._pipeline = mock_pipeline

    out_file = tmp_path / "out.mp4"
    response = provider.generate(dummy_request, str(out_file))

    assert response.output_path == str(out_file)
    mock_pipeline.generate_and_save.assert_called_once()
