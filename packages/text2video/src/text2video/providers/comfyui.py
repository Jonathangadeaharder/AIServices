import copy
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
        prompt = copy.deepcopy(self.workflow_template)

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

        # Select output node with ordered preference: "115" (SaveImage) → "114" (CreateVideo).
        # Only accept nodes that returned a non-empty list of artifacts.
        output_node: str | None = None
        for candidate in ("115", "114"):
            if images.get(candidate):
                output_node = candidate
                break

        # If neither preferred node has data, fall back to first non-empty node.
        if output_node is None:
            for key, value in images.items():
                if value:
                    output_node = key
                    break

        if output_node is None:
            raise RuntimeError("ComfyUI workflow did not return any output.")

        # Save the first frame or all frames
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        actual_output_path = output_path

        if len(images[output_node]) == 1:
            # Single output (possibly a video or a single frame)
            with open(output_path, "wb") as f:
                f.write(images[output_node][0])
        else:
            # Multiple frames — always save as .png regardless of the requested output extension,
            # since individual frames are images, not videos.
            stem = Path(output_path).stem
            parent = Path(output_path).parent
            frame_paths: list[Path] = []
            for i, frame_data in enumerate(images[output_node]):
                frame_path = parent / f"{stem}_{i:04d}.png"
                with open(frame_path, "wb") as f:
                    f.write(frame_data)
                frame_paths.append(frame_path)

            # Report the first frame as the primary output path
            actual_output_path = str(frame_paths[0])

        return Text2VideoResponse(
            output_path=actual_output_path,
            metadata={
                "provider": "comfyui",
                "seed": seed,
                "num_frames": request.num_frames,
                "num_output_files": len(images[output_node]),
                "fps": request.fps,
            },
        )
