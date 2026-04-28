import json
import random
from pathlib import Path

from aiservices_core.comfyui import ComfyUIClient
from aiservices_core.providers import BaseProvider

from ..models import Text2VideoRequest, Text2VideoResponse


class ComfyUIProvider(BaseProvider):
    """Local text-to-video provider using ComfyUI.

    Wraps the Wan 2.2 text-to-video workflow (wan22_t2v_api.json) and
    communicates with a running ComfyUI server over WebSocket.

    The workflow uses a two-pass sampling approach:
    - Pass 1 (high noise model + LoRA): Initial denoising
    - Pass 2 (low noise model + LoRA): Refinement
    """

    def __init__(
        self,
        server_address: str = "127.0.0.1:8188",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.client = ComfyUIClient(server_address=server_address)

        workflow_path = Path(__file__).parent.parent / "workflows" / "wan22_t2v_api.json"
        with open(workflow_path) as f:
            self.workflow_template = json.load(f)

    def generate(self, request: Text2VideoRequest, output_path: str) -> Text2VideoResponse:
        prompt = json.loads(json.dumps(self.workflow_template))  # deep copy

        # Update positive prompt
        prompt["89"]["inputs"]["text"] = request.prompt

        # Update negative prompt
        prompt["72"]["inputs"]["text"] = request.negative_prompt

        # Update dimensions and frame count
        prompt["74"]["inputs"]["width"] = request.width
        prompt["74"]["inputs"]["height"] = request.height
        prompt["74"]["inputs"]["length"] = request.num_frames

        # Update steps (split across two passes)
        total_steps = request.num_inference_steps
        mid_step = total_steps // 2
        prompt["81"]["inputs"]["steps"] = total_steps
        prompt["81"]["inputs"]["end_at_step"] = mid_step
        prompt["78"]["inputs"]["steps"] = total_steps
        prompt["78"]["inputs"]["start_at_step"] = mid_step

        # Update seed
        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        prompt["81"]["inputs"]["noise_seed"] = seed

        # Update FPS
        prompt["114"]["inputs"]["fps"] = request.fps

        # Execute
        images = self.client.get_images(prompt)

        # The SaveImage node "115" will have the output frames
        # Also check for video output from CreateVideo node "114"
        output_node = "115"
        if output_node not in images or not images[output_node]:
            # Try any available output node
            if images:
                output_node = next(iter(images))
            else:
                raise RuntimeError("ComfyUI workflow did not return any output.")

        # Save the first frame or all frames
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if len(images[output_node]) == 1:
            # Single output (possibly a video or a single frame)
            with open(output_path, "wb") as f:
                f.write(images[output_node][0])
        else:
            # Multiple frames — save each one
            stem = Path(output_path).stem
            suffix = Path(output_path).suffix or ".png"
            parent = Path(output_path).parent
            for i, frame_data in enumerate(images[output_node]):
                frame_path = parent / f"{stem}_{i:04d}{suffix}"
                with open(frame_path, "wb") as f:
                    f.write(frame_data)

        return Text2VideoResponse(
            output_path=output_path,
            metadata={
                "provider": "comfyui",
                "seed": seed,
                "num_frames": request.num_frames,
                "fps": request.fps,
            },
        )
