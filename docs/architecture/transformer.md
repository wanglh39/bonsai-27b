# Transformer 架构

> **核心**：RoPE + KV cache + hybrid attention (75% linear / 25% full)，
> 所有 Linear 可一键切换为 `HadamardTernaryLinear`。

## 架构总览

真实 Bonsai 基于 Qwen3.6-27B：

```
MiniBonsai
  ├── embed: Embedding
  ├── blocks: n_layers × SimpleBlock
  │     ├── norm1: RMSNorm
  │     ├── attn: SimpleAttention (full) 或 LinearAttention (linear)
  │     ├── norm2: RMSNorm
  │     └── mlp: SwiGLUMLP
  │           ├── gate: Linear / HadamardTernaryLinear
  │           ├── up: Linear / HadamardTernaryLinear
  │           └── down: Linear / HadamardTernaryLinear
  ├── norm_f: RMSNorm
  └── lm_head: Linear / HadamardTernaryLinear
```

## 组件

### RMSNorm

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$$

Qwen 系标配，比 LayerNorm 省一次减均值。

### RoPE (Rotary Position Embedding)

对 $q, k$ 的每对维度 $(2i, 2i+1)$ 做旋转：

$$q' = q \cdot \cos(m\theta) + \text{rotate\_half}(q) \cdot \sin(m\theta)$$
$$\theta_i = \text{base}^{-2i/d}$$

比绝对位置编码更适合长序列外推。Qwen 系标配。

```python
from bonsai import RoPE

rope = RoPE(head_dim=64, max_seq_len=2048)
q_rot, k_rot = rope(q, k, offset=0)  # offset 用于 KV cache
```

### KV Cache

自回归生成时复用历史 $K, V$，避免重算。

每步只算新 token 的 qkv，把新 $k, v$ 拼到 cache 上：

```python
from bonsai import KVCache

cache = KVCache()
# prefill
logits = model(tokens, kv_caches=[cache1, cache2, ...])
# 生成（每步只输入 1 个 token）
for _ in range(n_new):
    logits = model(next_tok, kv_caches=[cache1, cache2, ...])
    next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
```

### SwiGLU MLP

$$\text{SwiGLU}(x) = \text{down}(\text{silu}(\text{gate}(x)) \cdot \text{up}(x))$$

三层结构：gate、up 升维到 intermediate，down 降回 dim。Qwen/Llama 系标准激活。

### Hybrid Attention

75% 的层用 **linear attention**（$O(n)$），25% 用 **full self-attention**（$O(n^2)$）。

#### Full self-attention

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V$$

带 RoPE + KV cache。质量最好，但 $O(n^2)$ 复杂度。

#### Linear attention

用 ELU+1 核近似 softmax：

$$\phi(x) = \text{elu}(x) + 1$$
$$\text{Attn}(q, k, v) = \frac{\phi(q) \cdot (\phi(k)^\top \cdot v)}{\phi(q) \cdot \phi(k)^\top \cdot \mathbf{1}}$$

不需要显式 attention matrix，$O(n)$ 复杂度。省显存 + 长上下文快。

劣势：近似质量不如 softmax，长序列上信息会衰减。

#### 为什么 75/25

- 75% linear：大部分层用 linear，省显存 + 长上下文快
- 25% full：保留关键层的 full attention，确保信息聚合质量
- 这个配比是 Qwen3.6 的经验值

## 低 bit 切换

所有 Linear 可一键切换为 `HadamardTernaryLinear`：

```python
from bonsai import MiniBonsai

# FP16 版
float_model = MiniBonsai(vocab_size=1024, dim=256, n_layers=4, low_bit=False)

# 低 bit 版
low_bit_model = MiniBonsai(
    vocab_size=1024, dim=256, n_layers=4,
    low_bit=True,
    hybrid_attention=True,   # 75% linear / 25% full
    linear_ratio=0.75,
)

# 从 FP16 烧出低 bit 版
low_bit_model.prepare_from_float(float_model)
```

## generate（带 KV cache）

```python
# 自回归生成
generated = model.generate(tokens, n_new=50)  # (B, T+50)
```

内部用 KV cache：prefill 一次，之后每步只算 1 个 token。

## API

```python
from bonsai import (
    RMSNorm, RoPE, KVCache, SwiGLUMLP,
    SimpleAttention, LinearAttention, SimpleBlock, MiniBonsai,
)
```

---

详见 `bonsai/block.py`。