# text2video

Text-to-Video generation module for AIServices. Uses ComfyUI with Wan 2.2 as the local backend.

## Installation

This package is part of the AIServices monorepo. It can be installed directly via `uv`:

```bash
uv tool install ./packages/text2video
```

## Usage

```bash
text2video generate --prompt "A cinematic drone shot over a mountain lake" --output out.mp4
```

Available providers:
- `text2video.comfyui`: Uses a running ComfyUI server with the Wan 2.2 workflow (default)

## Prerequisites

For the `comfyui` provider, you need a running ComfyUI server at `127.0.0.1:8188` with the following models:

- `diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors`
- `diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors`
- `loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors`
- `loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors`
- `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- `vae/wan_2.1_vae.safetensors`
