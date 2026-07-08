# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Vendored from https://github.com/pytorch/torchtitan (main)

"""Minimal standalone Llama 3 model definition extracted from torchtitan.

This vendored copy removes all torchtitan infra dependencies (Trainer,
JobConfig, metrics, checkpointing) and keeps only the model architecture
so it can be used by self-contained benchmark scripts.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal config (replaces torchtitan.config / ModelSpec)
# ---------------------------------------------------------------------------

@dataclass
class ModelArgs:
    """Flat config that the vendored model constructor consumes."""

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    ffn_dim_multiplier: float | None = None
    multiple_of: int = 256
    vocab_size: int = 128256
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_seq_len: int = 2048
    enable_weight_tying: bool = False


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[: xq_.shape[-2]].unsqueeze(0).unsqueeze(0)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.n_rep = args.n_heads // args.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor | None = None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        xq, xk, xv = (t.transpose(1, 2) for t in (xq, xk, xv))
        xk, xv = (t.repeat_interleave(self.n_rep, dim=1) for t in (xk, xv))

        output = nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, is_causal=True if mask is None else False)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor | None = None):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        return h + self.feed_forward(self.ffn_norm(h))


# ---------------------------------------------------------------------------
# Full Llama model
# ---------------------------------------------------------------------------

class Llama3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(TransformerBlock(i, args) for i in range(args.n_layers))
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.enable_weight_tying:
            self.lm_head.weight = self.tok_embeddings.weight

        self.freqs_cis = precompute_freqs_cis(
            args.dim // args.n_heads, args.max_seq_len, theta=args.rope_theta
        )

    def forward(self, tokens: torch.Tensor):
        h = self.tok_embeddings(tokens)
        mask = torch.triu(torch.full((tokens.size(1), tokens.size(1)), float("-inf"), device=tokens.device), diagonal=1)
        freqs = self.freqs_cis.to(tokens.device)
        for layer in self.layers:
            h = layer(h, freqs, mask)
        h = self.norm(h)
        return self.lm_head(h)


# ---------------------------------------------------------------------------
# Config registry (llama3_dense_model_*) — truncated for benchmark
# ---------------------------------------------------------------------------

LLAMA3_CONFIGS: dict[str, dict] = {
    "debugmodel": {
        "dim": 256,
        "n_layers": 2,
        "n_heads": 16,
        "n_kv_heads": 8,
        "ffn_dim_multiplier": None,
        "multiple_of": 256,
        "vocab_size": 128256,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "max_seq_len": 2048,
        "enable_weight_tying": False,
    },
    "8b": {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,
        "ffn_dim_multiplier": 1.3,
        "multiple_of": 1024,
        "vocab_size": 128256,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "max_seq_len": 8192,
        "enable_weight_tying": False,
    },
}

LLAMA3_MODEL_SIZES = list(LLAMA3_CONFIGS.keys())


def build_llama3_model(model_size: str) -> Llama3Model:
    if model_size not in LLAMA3_CONFIGS:
        raise ValueError(f"Unknown model size '{model_size}'. Choices: {LLAMA3_MODEL_SIZES}")
    args = ModelArgs(**LLAMA3_CONFIGS[model_size])
    return Llama3Model(args)
