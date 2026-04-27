import json
import logging
import urllib.parse
import urllib.request
import uuid
from typing import Any

import websocket

logger = logging.getLogger(__name__)


class ComfyUIClient:
    """Client for executing workflows on a ComfyUI server."""

    def __init__(self, server_address: str = "127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, prompt: dict[str, Any], prompt_id: str) -> None:
        p = {"prompt": prompt, "client_id": self.client_id, "prompt_id": prompt_id}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        urllib.request.urlopen(req).read()

    def get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            return response.read()

    def get_history(self, prompt_id: str) -> dict[str, Any]:
        with urllib.request.urlopen(
            f"http://{self.server_address}/history/{prompt_id}"
        ) as response:
            return json.loads(response.read())

    def get_images(self, prompt: dict[str, Any]) -> dict[str, list[bytes]]:
        prompt_id = str(uuid.uuid4())
        self.queue_prompt(prompt, prompt_id)

        ws = websocket.WebSocket()
        ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")

        output_images = {}
        try:
            while True:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message["type"] == "executing":
                        data = message["data"]
                        if data["node"] is None and data["prompt_id"] == prompt_id:
                            break  # Execution is done
                else:
                    # binary data (latent previews)
                    continue

            history = self.get_history(prompt_id).get(prompt_id, {})
            outputs = history.get("outputs", {})

            for node_id, node_output in outputs.items():
                images_output = []
                if "images" in node_output:
                    for image in node_output["images"]:
                        image_data = self.get_image(
                            image["filename"], image["subfolder"], image["type"]
                        )
                        images_output.append(image_data)
                if images_output:
                    output_images[node_id] = images_output
        finally:
            ws.close()

        return output_images
