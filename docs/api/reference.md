# API 参考 —— bonsai 包完整文档

> **主线**：`bonsai/` 包是 Bonsai-27B 的核心算子库，分 6 个模块：
> `hadamard`（旋转）、`quantize`（量化）、`ops`（低 bit matmul）、`kernel`（硬件层）、`dspark`（推理加速）、`block`（架构）。
> 所有算子可**独立使用**，也可组合成完整模型。
> 本文按模块列出完整 API：每个函数/类的**签名、参数、返回值、数学公式、使用示例**。
> 所有 API 都从顶层包导出：`from bonsai import ...`。

## 1. bonsai.hadamard —— Hadamard 旋转

Hadamard 旋转是 Bonsai 不塌陷的支点。详见 [Hadamard 旋转](../operators/hadamard.md)。

### `hadamard_matrix(n, dtype=torch.float32)`

构造 $n$ 阶 Walsh-Hadamard 矩阵（$n$ 必须是 2 的幂）。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `n` | int | — | 矩阵阶数，必须是 2 的幂 |
| `dtype` | torch.dtype | float32 | 输出 dtype |

**返回**：$(n, n)$ 的 $\pm 1$ 正交矩阵（未归一化），满足 $H \cdot H^\top = n \cdot I$。

```python
from bonsai import hadamard_matrix
H = hadamard_matrix(8)  # (8, 8), ±1
assert torch.allclose(H @ H.T, 8 * torch.eye(8))
```

### `random_hadamard(n, generator=None)`

随机化 Hadamard：$H \cdot \text{diag}(\pm 1)$。打散 Walsh-Hadamard 的固定结构。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `n` | int | — | 矩阵阶数，必须是 2 的幂 |
| `generator` | torch.Generator | None | 随机数生成器 |

**返回**：$(n, n)$ 的随机化 Hadamard 矩阵。

### `BlockHadamard(block_size=128, randomized=True)`

分块 Hadamard 旋转器。对权重的每个 `block_size × block_size` 块做旋转。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `block_size` | int | 128 | 分块大小 |
| `randomized` | bool | True | 是否随机化 |

**方法**：

| 方法 | 用途 | 公式 |
|---|---|---|
| `rotate_weight(W)` | 离线旋转权重 | $W' = H_{\text{blk}} \cdot W \cdot H_{\text{blk}}^\top$ |
| `rotate_activation(x)` | 在线旋转 activation | $x' = x \cdot H_{\text{blk}}$ |
| `rotate_output(y)` | 在线反旋转输出 | $y = y' \cdot H_{\text{blk}}^\top$ |

```python
from bonsai import BlockHadamard
rot = BlockHadamard(block_size=128)
W_rot = rot.rotate_weight(W)    # 离线
x_rot = rot.rotate_activation(x)  # 在线
y = rot.rotate_output(y_rot)    # 在线
```

## 2. bonsai.quantize —— 三值化 / 二值化

三值化 $\{-1, 0, +1\}$ + group scale。详见 [Ternary quantization](../operators/ternary.md)。

### `ternarize(W, group_size=128, threshold_ratio=0.7)`

三值化权重到 $\{-1, 0, +1\}$ + group scale。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `W` | Tensor | — | 输入权重，任意形状 |
| `group_size` | int | 128 | group scale 的组大小 |
| `threshold_ratio` | float | 0.7 | 三值化阈值比例 |

**返回**：`(trit: int8, scale: float16)`

**赋值规则**：

$$
\tilde{w} = \begin{cases} +1 & w > \tau \cdot \alpha_g \\ -1 & w < -\tau \cdot \alpha_g \\ 0 & \text{otherwise} \end{cases}
$$

其中 $\alpha_g = \text{mean}(|w_g|)$ 是第 $g$ 组的 scale，$\tau$ 是 threshold_ratio。

```python
from bonsai import ternarize, dequantize
W = torch.randn(512, 512)
trit, scale = ternarize(W, group_size=128)
W_q = dequantize(trit, scale)  # 近似还原
```

### `binarize(W, group_size=128)`

二值化权重到 $\{-1, +1\}$ + group scale。

**参数**：同 `ternarize`（无 threshold_ratio）。

**返回**：`(sign: int8, scale: float16)`

**赋值规则**：$\tilde{w} = \text{sign}(w)$，幅度全交给 scale。

### `dequantize(trit_or_sign, scale, group_size=128)`

反量化：$\hat{w} = \tilde{w} \cdot \alpha_g$。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `trit_or_sign` | Tensor | — | 量化后的 int8 权重 |
| `scale` | Tensor | — | group scale |
| `group_size` | int | 128 | 组大小 |

**返回**：反量化后的 float 权重。

## 3. bonsai.ops —— 低 bit matmul

低 bit matmul 算子 + 完整的 HadamardTernaryLinear 层。

### `ternary_linear(x, trit, scale, group_size=128)`

低 bit 三值化 matmul：$y = x \cdot (\text{trit} \cdot \text{scale})^\top$。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `x` | Tensor | — | activation, `(..., in_features)` |
| `trit` | Tensor | — | 三值化权重 int8, `(out_features, in_features)` |
| `scale` | Tensor | — | group scale |
| `group_size` | int | 128 | 组大小 |

**返回**：`(..., out_features)`

**教学语义**：trit 是 int8（真实打包是 5-trit-per-byte 或 2-bit-slot），这里用 float 桥接让 PyTorch 能算。真实 kernel 路径：x 按 group 切片 → 对每个 group 做 trit matmul（可用加减/跳零）→ 乘 scale → 跨 group 求和。

### `binary_linear(x, sign, scale, group_size=128)`

低 bit 二值化 matmul：$y = x \cdot (\text{sign} \cdot \text{scale})^\top$。

**参数**：同 `ternary_linear`。

**真实 1-bit kernel 最快**：64 个 sign 塞进 int64，x 也符号化后用 XNOR+popcount 数一致位，一次 int64 运算顶 64 次乘加。

### `HadamardTernaryLinear(in_features, out_features, ...)`

完整的 Hadamard 旋转 + 三值化 Linear 层。这是 Bonsai 每一个 Linear 的最小完整复现。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `in_features` | int | — | 输入维度 |
| `out_features` | int | — | 输出维度 |
| `group_size` | int | 128 | 三值化的 group 大小（Bonsai g128） |
| `block_size` | int | 128 | Hadamard 分块大小 |
| `mode` | str | "ternary" | "ternary" 或 "binary" |
| `bias` | bool | True | 是否有 bias |

**方法**：

| 方法 | 用途 |
|---|---|
| `prepare_from_float(linear)` | 从 FP16 nn.Linear 离线量化 |
| `forward(x)` | 旋转 activation → 低 bit 乘 → 反旋转 |

**生命周期**：

离线（`prepare_from_float`）：

$$
W' = H_{\text{blk}} \cdot W \cdot H_{\text{blk}}^\top \quad \text{（旋转）}
$$
$$
\text{trit}, \text{scale} = \text{ternarize}(W') \quad \text{（三值化）}
$$

在线（`forward`）：

$$
y = \text{rotate\_output}(\text{ternary\_matmul}(\text{rotate\_activation}(x), \text{trit}, \text{scale}))
$$

```python
from bonsai import HadamardTernaryLinear
import torch.nn as nn

float_linear = nn.Linear(512, 512)
layer = HadamardTernaryLinear(512, 512, mode="ternary")
layer.prepare_from_float(float_linear)
y = layer(x)  # 前向走低 bit 路径
```

## 4. bonsai.kernel —— XNOR-popcount + packing

1-bit 硬件层。详见 [XNOR-popcount](../operators/xnor-popcount.md) 和 [packing](../operators/packing.md)。

### `xnor_popcount_matmul(a, b)`

1-bit XNOR-popcount matmul。$a, b \in \{-1, +1\}$。数学等价于浮点 matmul。

**参数**：

| 参数 | 类型 | 说明 |
|---|---|---|
| `a` | Tensor | `(m, k)`, $\in \{-1, +1\}$ |
| `b` | Tensor | `(k, n)`, $\in \{-1, +1\}$ |

**返回**：`(m, n)` float32

**原理**：

$$
a_{ik} \cdot b_{kj} = \begin{cases} +1 & a_{ik} = b_{kj} \\ -1 & a_{ik} \neq b_{kj} \end{cases}
$$

$$
\text{dot} = 2 \cdot \text{popcount}(\text{XNOR}(a, b)) - k
$$

```python
from bonsai import xnor_popcount_matmul
a = torch.sign(torch.randn(4, 8))  # ±1
b = torch.sign(torch.randn(8, 4))  # ±1
c = xnor_popcount_matmul(a, b)     # 等价于 a @ b
assert torch.allclose(c, a.float() @ b.float())
```

### `ternary_matmul_via_xnor(a, b)`

三值化 matmul via XNOR + mask。$a \in \{-1, +1\}$, $b \in \{-1, 0, +1\}$。

**参数**：

| 参数 | 类型 | 说明 |
|---|---|---|
| `a` | Tensor | `(m, k)`, $\in \{-1, +1\}$ |
| `b` | Tensor | `(k, n)`, $\in \{-1, 0, +1\}$ |

**返回**：`(m, n)` float32

**原理**：$b$ 有 0，不能纯 XNOR。拆成 nonzero_mask + sign，只在非零位置做 XNOR-popcount。额外开销：mask 计算 + gather（不规则内存访问）、每列非零数不同（load imbalance）、存 magnitude（额外 bit）。

### `packing_bpw()`

返回不同 packing 的 bpw 字典。

**返回**：

```python
{
    "1-bit (binary)": 1.125,    # 1 + 16/128
    "PTQ1_0 (dense trit)": 1.725,  # 8/5 + 16/128
    "PQ2_0 (2-bit slot)": 2.125,   # 2 + 16/128
}
```

### `pack_trits_ptq10(trits)` / `unpack_trits_ptq10(packed)`

PTQ1_0 packing：5 trit → 8 bit（base-3 编码）。

**`pack_trits_ptq10`**：

| 参数 | 类型 | 说明 |
|---|---|---|
| `trits` | Tensor | `(...,)` int8, $\in \{-1, 0, +1\}$，长度必须是 5 的倍数 |

**返回**：`(.../5,)` uint8

**`unpack_trits_ptq10`**：

| 参数 | 类型 | 说明 |
|---|---|---|
| `packed` | Tensor | `(...,)` uint8 |

**返回**：`(...*5,)` int8, $\in \{-1, 0, +1\}$

```python
from bonsai import pack_trits_ptq10, unpack_trits_ptq10
trits = torch.randint(-1, 2, (40,))
packed = pack_trits_ptq10(trits)       # uint8, 8 bytes
unpacked = unpack_trits_ptq10(packed)  # int8, 40 个
assert (trits == unpacked).all()       # 精确还原
```

### `pack_trits_pq20(trits)` / `unpack_trits_pq20(packed)`

PQ2_0 packing：4 trit → 8 bit（2-bit slot）。

**`pack_trits_pq20`**：

| 参数 | 类型 | 说明 |
|---|---|---|
| `trits` | Tensor | `(...,)` int8, $\in \{-1, 0, +1\}$，长度必须是 4 的倍数 |

**返回**：`(.../4,)` uint8

**`unpack_trits_pq20`**：

| 参数 | 类型 | 说明 |
|---|---|---|
| `packed` | Tensor | `(...,)` uint8 |

**返回**：`(...*4,)` int8, $\in \{-1, 0, +1\}$

## 5. bonsai.dspark —— DSpark drafter

DSpark block-denoising drafter。详见 [DSpark](../operators/dspark.md)。

### `TargetModel(vocab=100, dim=64)`

大模型（target）：教学用 embedding + GRU + linear。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `vocab` | int | 100 | 词表大小 |
| `dim` | int | 64 | 隐藏维度 |

**结构**：`Embedding(vocab, dim)` → `GRU(dim, dim)` → `Linear(dim, vocab)`

### `AutoregressiveDrafter(vocab=100, dim=32)`

传统自回归 drafter。

**方法**：

| 方法 | 用途 |
|---|---|
| `draft(prefix, k)` | 串行草 $k$ 个 token（$k$ 次前向） |

### `BlockDenoisingDrafter(vocab=100, dim=32, block_size=4)`

DSpark block-parallel drafter。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `vocab` | int | 100 | 词表大小 |
| `dim` | int | 32 | 隐藏维度 |
| `block_size` | int | 4 | block 大小 $k$ |

**结构**：`Embedding(vocab, dim)` → `GRU(dim, dim)` → `Linear(dim, vocab * block_size)`

**方法**：

| 方法 | 用途 |
|---|---|
| `draft(prefix, k)` | 一次前向草 $k$ 个 token（1 次前向） |

### `verify(target, prefix, drafted)`

验证：target 逐个检查 drafted token 是否匹配自己的 argmax。

**参数**：

| 参数 | 类型 | 说明 |
|---|---|---|
| `target` | TargetModel | 大模型 |
| `prefix` | Tensor | 前缀 token |
| `drafted` | Tensor | drafted token |

**返回**：`int`，接受数（从第一个不匹配位置截断）

**无损保证**：accepted token 和普通生成不可区分。

### `train_drafter(drafter, target, data, steps=200)`

训练 drafter 预测真实序列 data 的后续 token。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `drafter` | nn.Module | — | 要训练的 drafter |
| `target` | TargetModel | — | target 模型 |
| `data` | Tensor | — | 训练序列 |
| `steps` | int | 200 | 训练步数 |

block-denoising drafter 一次预测 $k$ 个真实 token（block-denoising objective）。自回归 drafter 逐个预测下一个真实 token。

### `make_sequence(kind, length=200, vocab=100)`

造 code-like（高可预测）或 chat-like（低可预测）序列。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `kind` | str | — | "code" 或 "chat" |
| `length` | int | 200 | 序列长度 |
| `vocab` | int | 100 | 词表大小 |

**返回**：`(length,)` int64 token 序列

- **code-like**：重复 pattern + 少量 noise（模拟代码的语法约束）→ 下一个 token 高度可预测
- **chat-like**：随机 + 少量 pattern（模拟自然语言的多样性）→ 下一个 token 不确定性强

```python
from bonsai import make_sequence
code_seq = make_sequence("code", 500, vocab=100)
chat_seq = make_sequence("chat", 500, vocab=100)
```

## 6. bonsai.block —— Transformer 架构

完整 Transformer 架构。详见 [Transformer 架构](../architecture/transformer.md)。

### `RMSNorm(dim, eps=1e-6)`

RMSNorm：$x / \sqrt{\text{mean}(x^2) + \epsilon} \cdot \gamma$。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `dim` | int | — | 维度 |
| `eps` | float | 1e-6 | 防除零 |

### `RoPE(head_dim, max_seq_len=2048, base=10000)`

Rotary Position Embedding。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `head_dim` | int | — | 每个头的维度 |
| `max_seq_len` | int | 2048 | 最大序列长度 |
| `base` | float | 10000 | 频率基数 |

**方法**：`forward(q, k, offset=0)` → `(q_rot, k_rot)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `q` | Tensor | `(B, n_heads, T, head_dim)` |
| `k` | Tensor | `(B, n_heads, T, head_dim)` |
| `offset` | int | KV cache 的起始位置 |

### `KVCache()`

KV cache。

**方法**：

| 方法 | 用途 |
|---|---|
| `update(k, v)` | 拼接新 k, v，返回全量 k, v |
| `clear()` | 清空 cache |
| `length` | property，当前 cache 长度 |

### `SwiGLUMLP(dim, intermediate, low_bit=False, ...)`

SwiGLU MLP：$\text{down}(\text{silu}(\text{gate}(x)) \cdot \text{up}(x))$。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `dim` | int | — | 输入/输出维度 |
| `intermediate` | int | — | 中间维度 |
| `low_bit` | bool | False | 是否用 HadamardTernaryLinear |

**方法**：`prepare_from_float(float_mlp)` 从 FP16 MLP 烧出低 bit 版。

### `SimpleAttention(dim, n_heads, low_bit=False, use_rope=True, max_seq_len=2048, ...)`

Full self-attention with RoPE + KV cache。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `dim` | int | — | 维度 |
| `n_heads` | int | — | 头数 |
| `low_bit` | bool | False | 是否用 HadamardTernaryLinear |
| `use_rope` | bool | True | 是否用 RoPE |
| `max_seq_len` | int | 2048 | 最大序列长度 |

**方法**：`prepare_from_float(float_attn)` 从 FP16 attention 烧出低 bit 版。

### `LinearAttention(dim, n_heads, low_bit=False, ...)`

Linear attention via ELU+1 核。$O(n)$ 复杂度。

**参数**：同 `SimpleAttention`（无 use_rope、max_seq_len）。

**数学**：

$$
\phi(x) = \text{elu}(x) + 1, \quad \text{Attn}(q, k, v) = \frac{\phi(q) \cdot (\phi(k)^\top \cdot v)}{\phi(q) \cdot \phi(k)^\top \cdot \mathbf{1}}
$$

### `SimpleBlock(dim, n_heads, intermediate, attention_type="full", ...)`

Transformer block：RMSNorm → Attn → 残差 → RMSNorm → MLP → 残差。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `dim` | int | — | 维度 |
| `n_heads` | int | — | 头数 |
| `intermediate` | int | — | MLP 中间维度 |
| `attention_type` | str | "full" | "full" 或 "linear" |
| `low_bit` | bool | False | 是否用低 bit |
| `use_rope` | bool | True | 是否用 RoPE |
| `max_seq_len` | int | 2048 | 最大序列长度 |

**方法**：`prepare_from_float(float_block)` 从 FP16 block 烧出低 bit 版。

### `MiniBonsai(...)`

完整模型。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `vocab_size` | int | 1024 | 词表大小 |
| `dim` | int | 256 | 隐藏维度 |
| `n_layers` | int | 4 | 层数 |
| `n_heads` | int | 8 | 注意力头数 |
| `intermediate` | int | 768 | MLP 中间维度 |
| `low_bit` | bool | False | 是否用 HadamardTernaryLinear |
| `hybrid_attention` | bool | False | 是否混合 linear/full attention |
| `linear_ratio` | float | 0.75 | linear attention 占比 |
| `use_rope` | bool | True | 是否用 RoPE |
| `max_seq_len` | int | 2048 | 最大序列长度 |

**方法**：

| 方法 | 用途 |
|---|---|
| `prepare_from_float(float_model)` | 从 FP16 模型烧出低 bit 版 |
| `forward(tokens, kv_caches=None)` | 前向，返回 logits |
| `generate(tokens, n_new=10)` | 带 KV cache 的自回归生成 |

**hybrid attention 布局**：前 `n_layers * linear_ratio` 层用 linear attention，其余用 full。

```python
from bonsai import MiniBonsai

# FP16 版
float_model = MiniBonsai(vocab_size=1024, dim=256, n_layers=4, low_bit=False)

# 低 bit 版
model = MiniBonsai(
    vocab_size=1024, dim=256, n_layers=4,
    low_bit=True,
    hybrid_attention=True,
    linear_ratio=0.75,
)
model.prepare_from_float(float_model)

# 生成
tokens = torch.randint(0, 1024, (1, 10))
generated = model.generate(tokens, n_new=50)  # (1, 60)
```

## 7. 导入速查

```python
# 一次性导入所有 API
from bonsai import (
    # hadamard
    hadamard_matrix, random_hadamard, BlockHadamard,
    # quantize
    ternarize, binarize, dequantize,
    # ops
    ternary_linear, binary_linear, HadamardTernaryLinear,
    # kernel
    xnor_popcount_matmul, ternary_matmul_via_xnor,
    packing_bpw, pack_trits_ptq10, unpack_trits_ptq10,
    pack_trits_pq20, unpack_trits_pq20,
    # dspark
    TargetModel, AutoregressiveDrafter, BlockDenoisingDrafter,
    verify, train_drafter, make_sequence,
    # block
    RMSNorm, RoPE, KVCache, SwiGLUMLP,
    SimpleAttention, LinearAttention, SimpleBlock, MiniBonsai,
)
```

## 8. 进一步阅读

- **[Hadamard 旋转](../operators/hadamard.md)**：`bonsai.hadamard` 的原理
- **[Ternary quantization](../operators/ternary.md)**：`bonsai.quantize` 的原理
- **[XNOR-popcount](../operators/xnor-popcount.md)**：`bonsai.kernel` 的原理
- **[packing](../operators/packing.md)**：`bonsai.kernel` 的 packing 格式
- **[DSpark](../operators/dspark.md)**：`bonsai.dspark` 的原理
- **[Transformer 架构](../architecture/transformer.md)**：`bonsai.block` 的原理
- **[实验解读](../experiments/guide.md)**：怎么用这些 API 做实验