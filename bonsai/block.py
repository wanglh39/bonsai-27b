"""Transformer block —— Bonsai 的模型架构层。

真实 Bonsai 基于 Qwen3.6-27B：
  - hybrid attention (75% linear / 25% full)
  - SwiGLU MLP
  - RoPE (Rotary Position Embedding)
  - RMSNorm
  - KV cache（推理加速）

本包提供两级架构：
  - SimpleAttention + SimpleBlock + MiniBonsai：基础版（full attention + RoPE + KV cache）
  - LinearAttention + hybrid_attention=True：完整版（hybrid 75% linear / 25% full）

所有 Linear 都能换成 HadamardTernaryLinear，对比 float vs 低 bit 的前向输出差异。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ops import HadamardTernaryLinear


class RMSNorm(nn.Module):
    """RMSNorm：x / sqrt(mean(x^2) + eps) * weight。Qwen 系标配，比 LayerNorm 省一次减均值。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class RoPE(nn.Module):
    """Rotary Position Embedding。

    对 q, k 的每对维度 (2i, 2i+1) 做旋转：
      q' = q * cos(m*θ) + rotate_half(q) * sin(m*θ)
      θ_i = base^(-2i/d)

    Qwen 系标配。比绝对位置编码更适合长序列外推。
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        half = head_dim // 2
        freqs = 1.0 / (base ** (torch.arange(0, half).float() / half))
        t = torch.arange(max_seq_len).float()
        angles = t.unsqueeze(1) * freqs.unsqueeze(0)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def _rotate_half(self, x: Tensor) -> Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, q: Tensor, k: Tensor, offset: int = 0) -> tuple[Tensor, Tensor]:
        """q, k: (B, n_heads, T, head_dim)。offset: KV cache 的起始位置。"""
        T = q.shape[2]
        cos = self.cos[offset:offset + T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[offset:offset + T].unsqueeze(0).unsqueeze(0)
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class KVCache:
    """KV cache：自回归生成时复用历史 K, V，避免重算。

    每步只算新 token 的 qkv，把新 k, v 拼到 cache 上。
    """

    def __init__(self):
        self.k: Tensor | None = None
        self.v: Tensor | None = None

    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """k, v: (B, n_heads, T_new, head_dim)。返回拼接后的全量 k, v。"""
        if self.k is None:
            self.k = k
            self.v = v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    def clear(self):
        self.k = None
        self.v = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP：down(silu(gate(x)) * up(x))。

    三层结构：gate、up 升维到 intermediate，down 降回 dim。
    SwiGLU = silu(gate) * up，是 Qwen/Llama 系的标准激活。
    """

    def __init__(self, dim: int, intermediate: int, low_bit: bool = False, **lb_kwargs):
        super().__init__()
        self.low_bit = low_bit
        if low_bit:
            self.gate = HadamardTernaryLinear(dim, intermediate, **lb_kwargs)
            self.up = HadamardTernaryLinear(dim, intermediate, **lb_kwargs)
            self.down = HadamardTernaryLinear(intermediate, dim, **lb_kwargs)
        else:
            self.gate = nn.Linear(dim, intermediate)
            self.up = nn.Linear(dim, intermediate)
            self.down = nn.Linear(intermediate, dim)

    @torch.no_grad()
    def prepare_from_float(self, float_mlp: "SwiGLUMLP") -> None:
        assert self.low_bit and not float_mlp.low_bit
        self.gate.prepare_from_float(float_mlp.gate)
        self.up.prepare_from_float(float_mlp.up)
        self.down.prepare_from_float(float_mlp.down)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SimpleAttention(nn.Module):
    """Full self-attention with RoPE + KV cache。

    qkv 三个投影 + o 输出投影，都可选低 bit。
    attention 用 softmax(QK^T / sqrt(d)) V。
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        low_bit: bool = False,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        **lb_kwargs,
    ):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.low_bit = low_bit

        if low_bit:
            self.qkv = HadamardTernaryLinear(dim, 3 * dim, **lb_kwargs)
            self.o = HadamardTernaryLinear(dim, dim, **lb_kwargs)
        else:
            self.qkv = nn.Linear(dim, 3 * dim)
            self.o = nn.Linear(dim, dim)

        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPE(self.head_dim, max_seq_len)

    @torch.no_grad()
    def prepare_from_float(self, float_attn: "SimpleAttention") -> None:
        assert self.low_bit and not float_attn.low_bit
        self.qkv.prepare_from_float(float_attn.qkv)
        self.o.prepare_from_float(float_attn.o)

    def forward(self, x: Tensor, kv_cache: KVCache | None = None) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        offset = 0
        if kv_cache is not None:
            offset = kv_cache.length
            k, v = kv_cache.update(k, v)

        if self.use_rope:
            q, k = self.rope(q, k, offset=offset)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.o(out)


class LinearAttention(nn.Module):
    """Linear attention via kernel trick —— O(n) 复杂度。

    用 ELU+1 核近似 softmax：
      φ(x) = elu(x) + 1
      attn(q,k,v) = (φ(q) @ (φ(k)^T @ v)) / (φ(q) @ (φ(k)^T @ 1))

    优势：不需要显式 attention matrix，O(n) 而非 O(n²)。
    劣势：近似质量不如 softmax，长序列上信息会衰减。

    Qwen3.6 的 hybrid 里 75% 层用这个，省显存 + 长上下文快。
    注意：linear attention 的 KV cache 语义是 running sum (S, z)，
    教学版暂不实现 linear 的 KV cache（kv_cache 参数被忽略）。
    """

    def __init__(self, dim: int, n_heads: int, low_bit: bool = False, **lb_kwargs):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.low_bit = low_bit

        if low_bit:
            self.qkv = HadamardTernaryLinear(dim, 3 * dim, **lb_kwargs)
            self.o = HadamardTernaryLinear(dim, dim, **lb_kwargs)
        else:
            self.qkv = nn.Linear(dim, 3 * dim)
            self.o = nn.Linear(dim, dim)

    @torch.no_grad()
    def prepare_from_float(self, float_attn: "LinearAttention") -> None:
        assert self.low_bit and not float_attn.low_bit
        self.qkv.prepare_from_float(float_attn.qkv)
        self.o.prepare_from_float(float_attn.o)

    def _phi(self, x: Tensor) -> Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: Tensor, kv_cache: KVCache | None = None) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self._phi(q)
        k = self._phi(k)

        attn_num = q @ (k.transpose(-2, -1) @ v)
        attn_den = q @ k.sum(dim=2, keepdim=True).transpose(-2, -1)
        out = attn_num / (attn_den + 1e-6)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.o(out)


class SimpleBlock(nn.Module):
    """一个 transformer block：RMSNorm → Attn → 残差 → RMSNorm → MLP → 残差。

    attention_type: "full"（SimpleAttention）或 "linear"（LinearAttention）。
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        intermediate: int,
        low_bit: bool = False,
        attention_type: str = "full",
        use_rope: bool = True,
        max_seq_len: int = 2048,
        **lb_kwargs,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        if attention_type == "linear":
            self.attn = LinearAttention(dim, n_heads, low_bit=low_bit, **lb_kwargs)
        else:
            self.attn = SimpleAttention(
                dim, n_heads, low_bit=low_bit,
                use_rope=use_rope, max_seq_len=max_seq_len, **lb_kwargs,
            )
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim, intermediate, low_bit=low_bit, **lb_kwargs)
        self.low_bit = low_bit
        self.attention_type = attention_type

    @torch.no_grad()
    def prepare_from_float(self, float_block: "SimpleBlock") -> None:
        assert self.low_bit and not float_block.low_bit
        self.norm1.weight.data.copy_(float_block.norm1.weight.data)
        self.norm2.weight.data.copy_(float_block.norm2.weight.data)
        self.attn.prepare_from_float(float_block.attn)
        self.mlp.prepare_from_float(float_block.mlp)

    def forward(self, x: Tensor, kv_cache: KVCache | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + self.mlp(self.norm2(x))
        return x


class MiniBonsai(nn.Module):
    """迷你 Bonsai：n_layers 个 SimpleBlock + embedding + LM head。

    hybrid_attention: True 时按 linear_ratio 比例混合 linear/full attention。
    Qwen3.6 用 75% linear / 25% full——前 75% 层用 linear，后 25% 用 full。
    所有 Linear 可一键切换为 HadamardTernaryLinear。
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        intermediate: int = 768,
        low_bit: bool = False,
        hybrid_attention: bool = False,
        linear_ratio: float = 0.75,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        **lb_kwargs,
    ):
        super().__init__()
        self.low_bit = low_bit
        self.dim = dim
        self.hybrid_attention = hybrid_attention
        self.embed = nn.Embedding(vocab_size, dim)

        blocks = []
        n_linear = int(n_layers * linear_ratio) if hybrid_attention else 0
        for i in range(n_layers):
            attn_type = "linear" if i < n_linear else "full"
            blocks.append(SimpleBlock(
                dim, n_heads, intermediate,
                low_bit=low_bit, attention_type=attn_type,
                use_rope=use_rope, max_seq_len=max_seq_len,
                **lb_kwargs,
            ))
        self.blocks = nn.ModuleList(blocks)
        self.norm_f = RMSNorm(dim)
        if low_bit:
            self.lm_head = HadamardTernaryLinear(dim, vocab_size, **lb_kwargs)
        else:
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    @torch.no_grad()
    def prepare_from_float(self, float_model: "MiniBonsai") -> None:
        """从一个 float MiniBonsai 烧出低 bit 版本。embedding/norm 直接拷。"""
        assert self.low_bit and not float_model.low_bit
        self.embed.weight.data.copy_(float_model.embed.weight.data)
        self.norm_f.weight.data.copy_(float_model.norm_f.weight.data)
        for lb, fb in zip(self.blocks, float_model.blocks):
            lb.prepare_from_float(fb)
        self.lm_head.prepare_from_float(float_model.lm_head)

    def forward(self, tokens: Tensor, kv_caches: list[KVCache] | None = None) -> Tensor:
        x = self.embed(tokens)
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, kv_cache=cache)
        x = self.norm_f(x)
        return self.lm_head(x)

    def generate(self, tokens: Tensor, n_new: int = 10) -> Tensor:
        """自回归生成（带 KV cache）。tokens: (B, T) → (B, T+n_new)。"""
        kv_caches = [KVCache() for _ in self.blocks]
        logits = self.forward(tokens, kv_caches=kv_caches)
        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = [next_tok]
        for _ in range(n_new - 1):
            logits = self.forward(next_tok, kv_caches=kv_caches)
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(next_tok)
        return torch.cat([tokens] + generated, dim=1)