import io

from aiservices_core.io import load_image, save_image
from PIL import Image


def test_save_image_creates_parent_dirs(tmp_path):
    img = Image.new("RGB", (32, 32))
    nested = tmp_path / "a" / "b" / "c" / "out.png"
    save_image(img, nested)
    assert nested.exists()


def test_load_image_from_path(tmp_path):
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    path = tmp_path / "test.png"
    img.save(path)
    loaded = load_image(str(path))
    assert loaded.size == (64, 64)


def test_load_image_from_url(mocker, tmp_path):
    mock_get = mocker.patch("requests.get")
    img = Image.new("RGB", (64, 64), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    mock_response = mocker.MagicMock()
    mock_response.content = buf.read()
    mock_response.raise_for_status = mocker.MagicMock()
    mock_get.return_value = mock_response

    loaded = load_image("https://example.com/test.png")
    assert loaded.size == (64, 64)
    mock_get.assert_called_once_with("https://example.com/test.png", timeout=10)


def test_load_image_from_http_url(mocker, tmp_path):
    mock_get = mocker.patch("requests.get")
    img = Image.new("RGB", (32, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    mock_response = mocker.MagicMock()
    mock_response.content = buf.read()
    mock_response.raise_for_status = mocker.MagicMock()
    mock_get.return_value = mock_response

    loaded = load_image("http://example.com/img.png")
    assert loaded.size == (32, 32)
