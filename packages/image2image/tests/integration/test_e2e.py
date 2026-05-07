from image2image.cli import app
from PIL import Image
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_end_to_end(tmp_path, mocker):
    def fake_load(self):
        self._model = {"stub": True}

    mocker.patch(
        "image2image.providers.mlx.MLXProvider._load_model",
        fake_load,
    )

    input_img = Image.new("RGB", (256, 128), color="red")
    input_path = tmp_path / "input.png"
    input_img.save(str(input_path))

    output_path = tmp_path / "out.png"

    result = runner.invoke(
        app,
        [
            "--input",
            str(input_path),
            "--prompt",
            "a red square",
            "--output",
            str(output_path),
            "--strength",
            "0.7",
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    output_img = Image.open(str(output_path))
    assert output_img.size == (256, 128)
