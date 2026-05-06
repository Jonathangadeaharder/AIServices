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
            "--input", "/tmp/test.png",
            "--prompt", "a test video",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0
    mock_registry.get.assert_called_once()
    mock_provider.generate.assert_called_once()


def test_generate_error(mocker, tmp_path):
    mock_registry = mocker.patch("image2video.cli.registry")
    mock_registry.get.side_effect = RuntimeError("failed")

    result = runner.invoke(
        app,
        ["--input", "/tmp/test.png", "--prompt", "test", "--output", "/tmp/out.mp4"],
    )
    assert result.exit_code == 1
    mock_registry.get.assert_called_once()


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
            "--input", "/tmp/test.png",
            "--prompt", "test",
            "--output", str(out),
            "--seconds", "4",
            "--fps", "24",
            "--width", "640",
            "--height", "640",
            "--steps", "4",
        ],
    )
    assert result.exit_code == 0
    call_args = mock_provider.generate.call_args
    req = call_args[0][0]
    assert req.num_frames == 96  # 4 seconds * 24 fps
    assert req.fps == 24
