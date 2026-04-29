# Copyright © 2024 Apple Inc.


import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from tqdm import tqdm

from .lora import LoRALinear
from .sampler import FluxSampler
from .utils import (
    load_ae,
    load_clip,
    load_clip_tokenizer,
    load_flow_model,
    load_t5,
    load_t5_tokenizer,
    resolve_model_name,
)


class FluxPipeline:
    def __init__(self, name: str, t5_padding: bool = True):
        self.dtype = mx.bfloat16
        self.name = resolve_model_name(name)
        self.t5_padding = t5_padding

        self.ae = load_ae(self.name)
        self.flow = load_flow_model(self.name)
        self.clip = load_clip(self.name)
        self.clip_tokenizer = load_clip_tokenizer(self.name)
        self.t5 = load_t5(self.name)
        self.t5_tokenizer = load_t5_tokenizer(self.name)
        self.sampler = FluxSampler(self.name)

    def ensure_models_are_loaded(self):
        mx.eval(
            self.ae.parameters(),
            self.flow.parameters(),
            self.clip.parameters(),
            self.t5.parameters(),
        )

    def reload_text_encoders(self):
        self.t5 = load_t5(self.name)
        self.clip = load_clip(self.name)

    def tokenize(self, text):
        t5_tokens = self.t5_tokenizer.encode(text, pad=self.t5_padding)
        clip_tokens = self.clip_tokenizer.encode(text)
        return t5_tokens, clip_tokens

    def _prepare_latent_images(self, x):
        b, h, w, c = x.shape

        x = x.reshape(b, h // 2, 2, w // 2, 2, c)
        x = x.transpose(0, 1, 3, 5, 2, 4).reshape(b, h * w // 4, c * 4)

        i = mx.zeros((h // 2, w // 2), dtype=mx.int32)
        j, k = mx.meshgrid(mx.arange(h // 2), mx.arange(w // 2), indexing="ij")
        x_ids = mx.stack([i, j, k], axis=-1)
        x_ids = mx.repeat(x_ids.reshape(1, h * w // 4, 3), b, 0)

        return x, x_ids

    def _prepare_conditioning(self, n_images, t5_tokens, clip_tokens):
        txt = self.t5(t5_tokens)
        if len(txt) == 1 and n_images > 1:
            txt = mx.broadcast_to(txt, (n_images, *txt.shape[1:]))
        txt_ids = mx.zeros((n_images, txt.shape[1], 3), dtype=mx.int32)

        vec = self.clip(clip_tokens).pooled_output
        if len(vec) == 1 and n_images > 1:
            vec = mx.broadcast_to(vec, (n_images, *vec.shape[1:]))

        return txt, txt_ids, vec

    def _denoising_loop(
        self,
        x_t,
        x_ids,
        txt,
        txt_ids,
        vec,
        num_steps: int = 35,
        guidance: float = 4.0,
        start: float = 1,
        stop: float = 0,
    ):
        B = len(x_t)

        def scalar(x):
            return mx.full((B,), x, dtype=self.dtype)

        guidance = scalar(guidance)
        timesteps = self.sampler.timesteps(
            num_steps,
            x_t.shape[1],
            start=start,
            stop=stop,
        )
        for i in range(num_steps):
            t = timesteps[i]
            t_prev = timesteps[i + 1]

            pred = self.flow(
                img=x_t,
                img_ids=x_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=scalar(t),
                guidance=guidance,
            )
            x_t = self.sampler.step(pred, x_t, t, t_prev)

            yield x_t

    def generate_latents(
        self,
        text: str,
        n_images: int = 1,
        num_steps: int = 35,
        guidance: float = 4.0,
        latent_size: tuple[int, int] = (64, 64),
        seed=None,
    ):
        if num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")
        if seed is not None:
            mx.random.seed(seed)

        x_T = self.sampler.sample_prior((n_images, *latent_size, 16), dtype=self.dtype)
        x_T, x_ids = self._prepare_latent_images(x_T)

        t5_tokens, clip_tokens = self.tokenize(text)
        txt, txt_ids, vec = self._prepare_conditioning(n_images, t5_tokens, clip_tokens)

        yield (x_T, x_ids, txt, txt_ids, vec)

        yield from self._denoising_loop(
            x_T, x_ids, txt, txt_ids, vec, num_steps=num_steps, guidance=guidance
        )

    def decode(self, x, latent_size: tuple[int, int] = (64, 64)):
        h, w = latent_size
        x = x.reshape(len(x), h // 2, w // 2, -1, 2, 2)
        x = x.transpose(0, 1, 4, 2, 5, 3).reshape(len(x), h, w, -1)
        x = self.ae.decode(x)
        return mx.clip(x + 1, 0, 2) * 0.5

    def generate_images(
        self,
        text: str,
        n_images: int = 1,
        num_steps: int = 35,
        guidance: float = 4.0,
        latent_size: tuple[int, int] = (64, 64),
        seed=None,
        reload_text_encoders: bool = True,
        progress: bool = True,
    ):
        latents = self.generate_latents(text, n_images, num_steps, guidance, latent_size, seed)
        mx.eval(next(latents))

        if reload_text_encoders:
            self.reload_text_encoders()

        for x_t in tqdm(latents, total=num_steps, disable=not progress, leave=True):
            mx.eval(x_t)

        images = []
        for i in tqdm(range(len(x_t)), disable=not progress, desc="generate images"):
            images.append(self.decode(x_t[i : i + 1], latent_size=latent_size))
            mx.eval(images[-1])
        images = mx.concatenate(images, axis=0)
        mx.eval(images)

        return images

    def training_loss(
        self,
        x_0: mx.array,
        t5_features: mx.array,
        clip_features: mx.array,
        guidance: mx.array,
    ):
        txt = t5_features
        txt_ids = mx.zeros(txt.shape[:-1] + (3,), dtype=mx.int32)
        vec = clip_features

        x_0, x_ids = self._prepare_latent_images(x_0)

        t = self.sampler.random_timesteps(*x_0.shape[:2], dtype=self.dtype)
        eps = mx.random.normal(x_0.shape, dtype=self.dtype)
        x_t = self.sampler.add_noise(x_0, t, noise=eps)
        x_t = mx.stop_gradient(x_t)

        pred = self.flow(
            img=x_t,
            img_ids=x_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t,
            guidance=guidance,
        )

        return (pred + x_0 - eps).square().mean()

    def linear_to_lora_layers(self, rank: int = 8, num_blocks: int = -1):
        all_blocks = self.flow.double_blocks + self.flow.single_blocks
        all_blocks.reverse()
        num_blocks = num_blocks if num_blocks > 0 else len(all_blocks)
        for _i, block in zip(range(num_blocks), all_blocks, strict=False):
            loras = []
            for name, module in block.named_modules():
                if isinstance(module, nn.Linear):
                    loras.append((name, LoRALinear.from_base(module, r=rank)))
            block.update_modules(tree_unflatten(loras))

    def fuse_lora_layers(self):
        fused_layers = []
        for name, module in self.flow.named_modules():
            if isinstance(module, LoRALinear):
                fused_layers.append((name, module.fuse()))
        self.flow.update_modules(tree_unflatten(fused_layers))
