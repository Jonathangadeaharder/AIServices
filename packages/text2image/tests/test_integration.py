"""Integration test for text2image module.

Verifies end-to-end: CLI -> provider -> image file.
Skipped if MLX/flux not available (CI without Apple Silicon).
"""

import pytest
from PIL import Image
from text2image.cli import app
from typer.testing import CliRunner

runner = CliRunner()


@pytest.mark.skipif(
    not pytest.importorskip("image2image.flux_mlx", reason="flux_mlx not available"),
    reason="MLX provider not available",
)
def test_generate_512x512_red_cube(tmp_path):
    """Render 512x512 image from 'a red cube', assert file exists + dimensions."""
    out = tmp_path / "cube.png"
    result = runner.invoke(
        app,
        [
            "--prompt",
            "a red cube",
            "--output",
            str(out),
            "--width",
            "512",
            "--height",
            "512",
            "--steps",
            "4",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert out.exists(), "Output file not created"
    assert out.stat().st_size > 0, "Output file is empty"

    img = Image.open(out)
    assert img.size == (512, 512), f"Wrong dimensions: {img.size}"
