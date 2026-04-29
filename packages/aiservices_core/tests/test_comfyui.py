import json

import pytest
from aiservices_core.comfyui import ComfyUIClient

FIXED_PROMPT_ID = "fixed-prompt-id"


def test_comfyui_client_init():
    client = ComfyUIClient(server_address="localhost:8188")
    assert client.server_address == "localhost:8188"
    assert client.client_id is not None


def test_queue_prompt(mocker):
    mock_urlopen = mocker.patch("aiservices_core.comfyui.urllib.request.urlopen")
    mock_response = mocker.MagicMock()
    mock_response.read.return_value = b"{}"
    mock_urlopen.return_value = mock_response

    client = ComfyUIClient()
    client.queue_prompt({"workflow": "test"}, "prompt-123")
    mock_urlopen.assert_called_once()


def test_get_history(mocker):
    mock_urlopen = mocker.patch("aiservices_core.comfyui.urllib.request.urlopen")
    mock_response = mocker.MagicMock()
    mock_response.read.return_value = json.dumps({"prompt-123": {"outputs": {}}}).encode()
    mock_response.__enter__ = mocker.MagicMock(return_value=mock_response)
    mock_response.__exit__ = mocker.MagicMock(return_value=False)
    mock_urlopen.return_value = mock_response

    client = ComfyUIClient()
    history = client.get_history("prompt-123")
    assert "prompt-123" in history


def test_get_image(mocker):
    mock_urlopen = mocker.patch("aiservices_core.comfyui.urllib.request.urlopen")
    mock_response = mocker.MagicMock()
    mock_response.read.return_value = b"fake-image-data"
    mock_response.__enter__ = mocker.MagicMock(return_value=mock_response)
    mock_response.__exit__ = mocker.MagicMock(return_value=False)
    mock_urlopen.return_value = mock_response

    client = ComfyUIClient()
    data = client.get_image("test.png", "subfolder", "output")
    assert data == b"fake-image-data"


def test_upload_image(mocker, tmp_path):
    mock_urlopen = mocker.patch("aiservices_core.comfyui.urllib.request.urlopen")
    mock_response = mocker.MagicMock()
    mock_response.read.return_value = b"{}"
    mock_urlopen.return_value = mock_response

    test_file = tmp_path / "test.png"
    test_file.write_bytes(b"\x89PNG\r\n\x1a\n")

    client = ComfyUIClient()
    client.upload_image(str(test_file), "test.png")
    mock_urlopen.assert_called_once()


def test_get_images_full_workflow(mocker):
    mocker.patch(
        "aiservices_core.comfyui.uuid.uuid4",
        return_value=mocker.MagicMock(__str__=lambda self: FIXED_PROMPT_ID),
    )
    mock_urlopen = mocker.patch("aiservices_core.comfyui.urllib.request.urlopen")
    mock_ws_cls = mocker.patch("aiservices_core.comfyui.websocket.WebSocket")

    ws_instance = mocker.MagicMock()
    mock_ws_cls.return_value = ws_instance

    executing_msg = json.dumps({
        "type": "executing",
        "data": {"node": None, "prompt_id": FIXED_PROMPT_ID},
    })
    ws_instance.recv.return_value = executing_msg

    history_data = {
        FIXED_PROMPT_ID: {
            "outputs": {
                "node1": {
                    "images": [{"filename": "out.png", "subfolder": "", "type": "output"}]
                }
            }
        }
    }

    call_count = 0

    def urlopen_side_effect(req, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_resp = mocker.MagicMock()
        if call_count <= 1:
            mock_resp.read.return_value = b"{}"
            return mock_resp
        elif call_count == 2:
            mock_resp.read.return_value = json.dumps(history_data).encode()
        else:
            mock_resp.read.return_value = b"png-bytes"
        mock_resp.__enter__ = mocker.MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = mocker.MagicMock(return_value=False)
        return mock_resp

    mock_urlopen.side_effect = urlopen_side_effect

    client = ComfyUIClient()
    result = client.get_images({"test": "workflow"})

    assert "node1" in result
    assert result["node1"] == [b"png-bytes"]
    ws_instance.connect.assert_called_once()
    ws_instance.close.assert_called_once()


def test_get_images_timeout(mocker):
    import websocket as ws_mod

    mocker.patch(
        "aiservices_core.comfyui.uuid.uuid4",
        return_value=mocker.MagicMock(__str__=lambda self: FIXED_PROMPT_ID),
    )
    mocker.patch("aiservices_core.comfyui.urllib.request.urlopen")
    mock_ws_cls = mocker.patch("aiservices_core.comfyui.websocket.WebSocket")

    ws_instance = mocker.MagicMock()
    ws_instance.recv.side_effect = ws_mod.WebSocketTimeoutException("timeout")
    mock_ws_cls.return_value = ws_instance

    client = ComfyUIClient()

    with pytest.raises(TimeoutError, match="timed out"):
        client.get_images({"test": "workflow"})

    ws_instance.close.assert_called_once()


def test_get_images_ignores_binary(mocker):
    mocker.patch(
        "aiservices_core.comfyui.uuid.uuid4",
        return_value=mocker.MagicMock(__str__=lambda self: FIXED_PROMPT_ID),
    )
    mock_urlopen = mocker.patch("aiservices_core.comfyui.urllib.request.urlopen")
    mock_ws_cls = mocker.patch("aiservices_core.comfyui.websocket.WebSocket")

    ws_instance = mocker.MagicMock()
    mock_ws_cls.return_value = ws_instance

    executing_msg = json.dumps({
        "type": "executing",
        "data": {"node": None, "prompt_id": FIXED_PROMPT_ID},
    })
    ws_instance.recv.side_effect = [b"binary-data", executing_msg]

    history_data = {FIXED_PROMPT_ID: {"outputs": {}}}

    mock_resp = mocker.MagicMock()
    mock_resp.read.return_value = json.dumps(history_data).encode()
    mock_resp.__enter__ = mocker.MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = mocker.MagicMock(return_value=False)
    mock_urlopen.return_value = mock_resp

    client = ComfyUIClient()
    result = client.get_images({"test": "workflow"})

    assert result == {}
