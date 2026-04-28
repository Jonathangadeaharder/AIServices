from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class MarianConfig:
    vocab_size: int = 56847
    d_model: int = 1024
    encoder_layers: int = 6
    decoder_layers: int = 6
    encoder_attention_heads: int = 16
    decoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    decoder_ffn_dim: int = 4096
    max_position_embeddings: int = 1024
    scale_embedding: bool = True
    decoder_start_token_id: int = 56846
    eos_token_id: int = 44507
    pad_token_id: int = 56846
    activation_function: str = "relu"
    share_encoder_decoder_embeddings: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MarianConfig:
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


def _map_weight_name(hf_name: str) -> str | None:
    if hf_name == "model.shared.weight":
        return "embed_tokens.weight"
    if hf_name == "model.decoder.embed_tokens.weight":
        return None
    if hf_name == "final_logits_bias":
        return None
    for prefix in ("model.encoder.", "model.decoder."):
        if hf_name.startswith(prefix):
            return hf_name.replace("model.", "", 1)
    if hf_name == "lm_head.weight":
        return hf_name
    return hf_name


class MarianAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        _, src_len, _ = key.shape
        q = (
            self.q_proj(query)
            .reshape(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(key)
            .reshape(batch_size, src_len, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(value)
            .reshape(batch_size, src_len, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        scale = mx.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale
        if mask is not None:
            scores = scores + mask
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.out_proj(out)


class MarianEncoderLayer(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()
        self.self_attn = MarianAttention(config.d_model, config.encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, x, mask=None):
        residual = x
        x = self.self_attn(x, x, x, mask)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        residual = x
        x = mx.relu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x


class MarianEncoder(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.d_model
        )
        self.layers = [MarianEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, x, mask=None):
        positions = mx.arange(x.shape[1])
        x = x + self.embed_positions(positions)
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)


class MarianDecoderLayer(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()
        self.self_attn = MarianAttention(config.d_model, config.decoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.encoder_attn = MarianAttention(
            config.d_model, config.decoder_attention_heads
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, x, enc_output, tgt_mask=None, memory_mask=None):
        residual = x
        x = self.self_attn(x, x, x, tgt_mask)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        residual = x
        x = self.encoder_attn(x, enc_output, enc_output, memory_mask)
        x = residual + x
        x = self.encoder_attn_layer_norm(x)
        residual = x
        x = mx.relu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x


class MarianDecoder(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.d_model
        )
        self.layers = [MarianDecoderLayer(config) for _ in range(config.decoder_layers)]
        self.layer_norm = nn.LayerNorm(config.d_model)

    def __call__(self, x, enc_output, tgt_mask=None, memory_mask=None):
        positions = mx.arange(x.shape[1])
        x = x + self.embed_positions(positions)
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        return self.layer_norm(x)


class MarianMTModel(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = MarianEncoder(config)
        self.decoder = MarianDecoder(config)

    def encode(self, input_ids):
        scale = self.config.d_model**0.5 if self.config.scale_embedding else 1.0
        x = self.embed_tokens(input_ids) * scale
        return self.encoder(x)

    def decode(self, decoder_input_ids, enc_output, tgt_mask=None):
        scale = self.config.d_model**0.5 if self.config.scale_embedding else 1.0
        x = self.embed_tokens(decoder_input_ids) * scale
        x = self.decoder(x, enc_output, tgt_mask)
        return x @ self.embed_tokens.weight.T

    def __call__(self, input_ids, decoder_input_ids):
        enc = self.encode(input_ids)
        tgt_mask = _causal_mask(decoder_input_ids.shape[1])
        return self.decode(decoder_input_ids, enc, tgt_mask)


def _causal_mask(seq_len: int):
    mask = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)
    return mask.reshape(1, 1, seq_len, seq_len)


def hf_from_pretrained(model_name: str):
    from transformers import MarianMTModel

    return MarianMTModel.from_pretrained(model_name)


def convert_hf_to_mlx(model_name: str | Path, output_dir: str | Path) -> None:
    model = hf_from_pretrained(str(model_name))
    config_dict = model.config.to_dict()

    weights = {}
    for name, param in model.named_parameters():
        mapped = _map_weight_name(name)
        if mapped is None:
            continue
        if isinstance(param, np.ndarray):
            weights[mapped] = param
        elif hasattr(param, "detach"):
            weights[mapped] = param.detach().cpu().numpy()
        else:
            weights[mapped] = np.asarray(param)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "weights.npz", **weights)

    cfg = MarianConfig.from_dict(config_dict)
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f)


def load_mlx_model(model_dir: str | Path) -> tuple[MarianMTModel, MarianConfig]:
    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        cfg = MarianConfig.from_dict(json.load(f))

    model = MarianMTModel(cfg)
    npz = np.load(model_dir / "weights.npz")
    weights = {k: mx.array(v) for k, v in npz.items()}
    model.update(weights)

    return model, cfg


def greedy_decode(
    model: MarianMTModel,
    input_ids,
    max_length: int = 512,
    eos_token_id: int = 44507,
    decoder_start_token_id: int = 56846,
):
    enc_output = model.encode(input_ids)
    batch_size = input_ids.shape[0]
    decoder_ids = mx.full((batch_size, 1), decoder_start_token_id, dtype=mx.int32)

    for _ in range(max_length - 1):
        tgt_mask = _causal_mask(decoder_ids.shape[1])
        logits = model.decode(decoder_ids, enc_output, tgt_mask)
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        decoder_ids = mx.concatenate([decoder_ids, next_token], axis=1)
        if (next_token.squeeze(-1) == eos_token_id).all():
            break

    return decoder_ids
