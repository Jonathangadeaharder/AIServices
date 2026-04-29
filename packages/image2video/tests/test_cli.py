from image2video.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_generate_success(mocker, tmp_path):
    mock_registry = mocker.patch("image2video.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.mp4")
    mock_response.metadata = {"provider": "mlx"}
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.mp4"
    result = runner.invoke(
        app,
        [
            "--image", "/tmp/test.png",
            "--prompt", "a test video",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0


def test_generate_error(mocker, tmp_path):
    mock_registry = mocker.patch("image2video.cli.registry")
    mock_registry.get.side_effect = RuntimeError("failed")

    result = runner.invoke(
        app,
        ["--image", "/tmp/test.png", "--prompt", "test", "--output", "/tmp/out.mp4"],
    )
    assert result.exit_code == 1


def test_generate_with_options(mocker, tmp_path):
    mock_registry = mocker.patch("image2video.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.mp4")
    mock_response.metadata = {"provider": "mlx"}
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.mp4"
    result = runner.invoke(
        app,
        [
            "--image", "/tmp/test.png",
            "--prompt", "test",
            "--output", str(out),
            "--width", "640",
            "--height", "640",
            "--frames", "81",
            "--steps", "4",
            "--fps", "16",
        ],
    )
    assert result.exit_code == 0
