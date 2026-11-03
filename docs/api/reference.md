# API 参考

> `bonsai/` 包的完整 API。所有算子可独立使用，也可组合成完整模型。

## bonsai.hadamard — Hadamard 旋转

### `hadamard_matrix(n, dtype=float32)`

构造 $n$ 阶 Walsh-Hadamard 矩阵（$n$ 必须是 2 的幂）。

**返回**：$\pm 1$ 的正交矩阵（未归一化），满足 $H \cdot H^\top = n \cdot I$。

### `random_hadamard(n, generator=None)`

随机化 Hadamard：$H \cdot \text{diag}(\pm 1)$。打散 Walsh-Hadamard 的固定结构。

### `BlockHadamard(block_size=128, randomized=True)`

分块 Hadamard 旋转器。

| 方法 | 用途 | 公式 |
|---|---|---|
| `rotate_weight(W)` | 离线 | $W' = H_{\text{blk}} \cdot W \cdot H_{\text{blk}}^\top$ |
| `rotate_activation(x)` | 在线 | $x' = x \cdot H_{\text{blk}}$ |
| `rotate_output(y)` | 在线 | $y = y' \cdot H_{\text{blk}}^\top$ |

## bonsai.quantize — 三值化 / 二值化

### `ternarize(W, group_size=128, threshold_ratio=0.7)`

三值化权重到 $\{−1, 0, +1\}$ + group scale。

**返回**：`(trit: int8, scale: float)`。

### `binarize(W, group_size=128)`

二值化权重到 $\{−1, +1\}$ + group scale。

**返回**：`(sign: int8, scale: float)`。

### `dequantize(trit_or_sign, scale, group_size=128)`

反量化：$\hat{w} = \tilde{w} \cdot \alpha_g$。

## bonsai.ops — 低 bit matmul

### `ternary_linear(x, trit, scale, group_size=128)`

低 bit 三值化 matmul：$y = x \cdot (\text{trit} \cdot \text{scale})^\top$。

### `binary_linear(x, sign, scale, group_size=128)`

低 bit 二值化 matmul。

### `HadamardTernaryLinear(in_features, out_features, ...)`

完整的 Hadamard 旋转 + 三值化 Linear 层。

| 方法 | 用途 |
|---|---|
| `prepare_from_float(linear)` | 从 FP16 Linear 离线量化 |
| `forward(x)` | 旋转 activation → 低 bit 乘 → 反旋转 |

```python
layer = HadamardTernaryLinear(512, 512, mode="ternary")
layer.prepare_from_float(float_linear)
y = layer(x)
```

## bonsai.kernel — XNOR-popcount + packing

### `xnor_popcount_matmul(a, b)`

1-bit XNOR-popcount matmul。$a, b \in \{−1, +1\}$。数学等价于浮点 matmul。

### `ternary_matmul_via_xnor(a, b)`

三值化 matmul via XNOR + mask。$a \in \{−1, +1\}$, $b \in \{−1, 0, +1\}$。

### `packing_bpw()`

返回不同 packing 的 bpw 字典：

```python
{
    "1-bit (binary)": 1.125,
    "PTQ1_0 (dense trit)": 1.725,
    "PQ2_0 (2-bit slot)": 2.125,
}
```

### `pack_trits_ptq10(trits)` / `unpack_trits_ptq10(packed)`

PTQ1_0 packing：5 trit → 8 bit（base-3 编码）。

### `pack_trits_pq20(trits)` / `unpack_trits_pq20(packed)`

PQ2_0 packing：4 trit → 8 bit（2-bit slot）。

## bonsai.dspark — DSpark drafter

### `TargetModel(vocab=100, dim=64)`

大模型（target）：embedding + GRU + linear。

### `AutoregressiveDrafter(vocab=100, dim=32)`

传统自回归 drafter。`draft(prefix, k)` 串行草 $k$ 个 token。

### `BlockDenoisingDrafter(vocab=100, dim=32, block_size=4)`

DSpark block-parallel drafter。`draft(prefix, k)` 一次前向草 $k$ 个 token。

### `verify(target, prefix, drafted)`

验证：返回接受数（从第一个不匹配位置截断）。无损。

### `train_drafter(drafter, target, data, steps=200)`

训练 drafter 预测真实序列 data 的后续 token。

### `make_sequence(kind, length=200, vocab=100)`

造 code-like（高可预测）或 chat-like（低可预测）序列。

## bonsai.block — Transformer 架构

### `RMSNorm(dim, eps=1e-6)`

RMSNorm：$x / \sqrt{\text{mean}(x^2) + \epsilon} \cdot \gamma$。

### `RoPE(head_dim, max_seq_len=2048, base=10000)`

Rotary Position Embedding。`forward(q, k, offset=0)` 返回旋转后的 $q, k$。

### `KVCache()`

KV cache。`update(k, v)` 返回拼接后的全量 $k, v$。`clear()` 清空。

### `SwiGLUMLP(dim, intermediate, low_bit=False)`

SwiGLU MLP：$\text{down}(\text{silu}(\text{gate}(x)) \cdot \text{up}(x))$。

### `SimpleAttention(dim, n_heads, low_bit=False, use_rope=True)`

Full self-attention with RoPE + KV cache。

### `LinearAttention(dim, n_heads, low_bit=False)`

Linear attention via ELU+1 核。$O(n)$ 复杂度。

### `SimpleBlock(dim, n_heads, intermediate, attention_type="full", ...)`

Transformer block。`attention_type`: `"full"` 或 `"linear"`。

### `MiniBonsai(...)`

完整模型。

| 参数 | 默认 | 说明 |
|---|---|---|
| `vocab_size` | 1024 | 词表大小 |
| `dim` | 256 | 隐藏维度 |
| `n_layers` | 4 | 层数 |
| `n_heads` | 8 | 注意力头数 |
| `intermediate` | 768 | MLP 中间维度 |
| `low_bit` | False | 是否用 HadamardTernaryLinear |
| `hybrid_attention` | False | 是否混合 linear/full attention |
| `linear_ratio` | 0.75 | linear attention 占比 |
| `use_rope` | True | 是否用 RoPE |

| 方法 | 用途 |
|---|---|
| `prepare_from_float(float_model)` | 从 FP16 模型烧出低 bit 版 |
| `forward(tokens, kv_caches=None)` | 前向 |
| `generate(tokens, n_new=10)` | 带 KV cache 的自回归生成 |

```python
from bonsai import MiniBonsai

model = MiniBonsai(low_bit=True, hybrid_attention=True)
model.prepare_from_float(float_model)
generated = model.generate(tokens, n_new=50)
```