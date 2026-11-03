# Transformer 架构 —— MiniBonsai 的完整实现

> **主线**：Bonsai 的算子（Hadamard 旋转、三值化、XNOR-popcount、packing）是**零件**，
> 但零件不等于模型——需要一个完整 Transformer 架构把它们组装起来，才能端到端对比 float vs 低 bit。
> 本文从 MiniBonsai 的整体结构出发，逐个讲清每个组件的**数学基础、实现细节、和低 bit 的关系**：
> RMSNorm 为什么比 LayerNorm 省、RoPE 怎么对每对维度做旋转、KV cache 怎么把 $O(n^2)$ 降到 $O(n)$、
> hybrid attention 的 75/25 配比是什么意思、SwiGLU 为什么是三层结构。
> 最后讲清 `prepare_from_float` 怎么从 FP16 模型一键烧出低 bit 版，以及 `generate` 的 KV cache 流程。

## 1. 架构总览

### 1.1 真实 Bonsai vs MiniBonsai

真实 Bonsai 基于 Qwen3.6-27B，有数十层、隐藏维度数千。MiniBonsai 是**教学缩微版**——保留所有关键结构，只缩小尺寸：

| | 真实 Bonsai | MiniBonsai |
|---|---|---|
| 基座 | Qwen3.6-27B | 4 层、dim=256 |
| 参数量 | 27B | ~4M |
| attention | hybrid 75% linear / 25% full | 同 |
| 权重 | ternary {−1,0,+1} + scale | 同 |
| 用途 | 端侧推理 | 算子验证 + 对比实验 |

### 1.2 MiniBonsai 的结构

```
MiniBonsai
  ├── embed: Embedding(vocab, dim)
  ├── blocks: n_layers × SimpleBlock
  │     ├── norm1: RMSNorm(dim)
  │     ├── attn: SimpleAttention (full) 或 LinearAttention (linear)
  │     │     ├── qkv: Linear / HadamardTernaryLinear(dim, 3*dim)
  │     │     ├── rope: RoPE(head_dim, max_seq_len)    [仅 full attention]
  │     │     └── o: Linear / HadamardTernaryLinear(dim, dim)
  │     ├── norm2: RMSNorm(dim)
  │     └── mlp: SwiGLUMLP(dim, intermediate)
  │           ├── gate: Linear / HadamardTernaryLinear(dim, intermediate)
  │           ├── up: Linear / HadamardTernaryLinear(dim, intermediate)
  │           └── down: Linear / HadamardTernaryLinear(intermediate, dim)
  ├── norm_f: RMSNorm(dim)
  └── lm_head: Linear / HadamardTernaryLinear(dim, vocab)
```

### 1.3 前向流程

```python
x = self.embed(tokens)                    # (B, T) → (B, T, dim)
for block in self.blocks:
    x = block(x, kv_cache=cache)          # 残 attention + MLP
x = self.norm_f(x)                        # 最终 norm
logits = self.lm_head(x)                  # (B, T, vocab)
```

每个 `SimpleBlock` 的内部：

```python
x = x + self.attn(self.norm1(x))          # pre-norm + attention + 残差
x = x + self.mlp(self.norm2(x))           # pre-norm + MLP + 残差
```

**Pre-norm 结构**（norm 在 attention/mlp 之前）——比 post-norm 更稳定，是现代 Transformer 的标配。

## 2. 组件详解

### 2.1 RMSNorm

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma
$$

其中 $\gamma$ 是可学习缩放，$\epsilon$ 是防除零的小常数。

**vs LayerNorm**：LayerNorm 要先减均值再除标准差：

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\text{var}(x) + \epsilon}} \cdot \gamma
$$

RMSNorm **省了一次减均值**——均值计算 + 减法。在长序列 + 大 batch 下这是可观的节省。

```python
rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
return x * rms * self.weight
```

**和低 bit 的关系**：RMSNorm 不做权重投影，只做逐元素缩放——**不需要量化**，保持 FP16 精度。量化只作用于 Linear 层（qkv、o、gate、up、down、lm_head）。

### 2.2 RoPE (Rotary Position Embedding)

RoPE 对 $q, k$ 的每对维度 $(2i, 2i+1)$ 做旋转，注入位置信息：

$$
q' = q \cdot \cos(m\theta) + \text{rotate\_half}(q) \cdot \sin(m\theta)
$$

$$
\theta_i = \text{base}^{-2i/d}, \quad m = \text{position}
$$

其中 `rotate_half` 把前半维和后半维交换并取反：

```python
def _rotate_half(self, x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)
```

**数学含义**：对每对维度 $(q_{2i}, q_{2i+1})$ 做 $m\theta_i$ 角度的 2D 旋转：

$$
\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix}
=
\begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}
\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
$$

**为什么用旋转而不是绝对位置编码？**

- 绝对位置编码（如 sinusoidal）把位置信息**加**到 embedding 上，长序列外推时未见过位置的编码可能失效
- RoPE 把位置信息**乘**到 $q, k$ 上，内积 $q_m^\top k_n$ 只依赖**相对位置** $m - n$，外推性更好

**和 KV cache 的配合**：`offset` 参数指定当前 cache 的起始位置，RoPE 从 `offset` 开始取 cos/sin：

```python
cos = self.cos[offset:offset + T]  # T 是当前 batch 的序列长度
```

这让生成时每步只对新 token 应用 RoPE，不用重算历史。

**和低 bit 的关系**：RoPE 只对 $q, k$ 做逐元素旋转，不涉及权重——**不需要量化**。

### 2.3 KV Cache

自回归生成时，每步只算新 token 的 $q, k, v$，把新 $k, v$ **拼到 cache** 上：

```python
class KVCache:
    def update(self, k, v):
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)  # 沿序列维拼接
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v
```

**复杂度改进**：

| | 无 cache | 有 cache |
|---|---|---|
| 第 $t$ 步 | 重算所有 $t$ 个 token 的 attention：$O(t^2 \cdot d)$ | 只算 1 个新 token vs $t$ 个历史：$O(t \cdot d)$ |
| 总复杂度（生成 $n$ 个 token） | $\sum_{t=1}^{n} O(t^2) = O(n^3)$ | $\sum_{t=1}^{n} O(t) = O(n^2)$ |

**和低 bit 的关系**：KV cache 存的是 $k, v$ **activation**，不是权重——activation 保持 FP16，不量化。量化只作用于权重。

### 2.4 SwiGLU MLP

$$
\text{SwiGLU}(x) = \text{down}(\text{silu}(\text{gate}(x)) \cdot \text{up}(x))
$$

其中 $\text{silu}(x) = x \cdot \sigma(x)$（SiLU 激活）。

三层结构：

1. `gate(x)`：$\text{dim} \to \text{intermediate}$，生成"门控信号"
2. `up(x)`：$\text{dim} \to \text{intermediate}$，生成"内容信号"
3. `down(...)`：$\text{intermediate} \to \text{dim}$，降回原维度

**为什么是 gate × up 而不是单个 Linear？**

GLU 结构让模型能**动态选择**信息：`silu(gate(x))` 是一个软门（0 到 1 的缩放），`up(x)` 是内容，相乘让模型能"按需放行"信息。这比单个 Linear + 激活表达力更强。

```python
def forward(self, x):
    return self.down(F.silu(self.gate(x)) * self.up(x))
```

**和低 bit 的关系**：三个 Linear（gate、up、down）都换成 `HadamardTernaryLinear`——这是低 bit 量化的主要目标。

### 2.5 Hybrid Attention

75% 的层用 **linear attention**（$O(n)$），25% 用 **full self-attention**（$O(n^2)$）。

#### Full self-attention

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

带 RoPE + KV cache。质量最好，但 $O(n^2)$ 复杂度——长序列时显存和计算量爆炸。

```python
attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
attn = attn.softmax(dim=-1)
out = attn @ v
```

#### Linear attention

用 ELU+1 核近似 softmax，避免显式 attention matrix：

$$
\phi(x) = \text{elu}(x) + 1
$$

$$
\text{Attn}(q, k, v) = \frac{\phi(q) \cdot (\phi(k)^\top \cdot v)}{\phi(q) \cdot \phi(k)^\top \cdot \mathbf{1}}
$$

关键：$\phi(k)^\top \cdot v$ 是一个 $(d \times d)$ 矩阵 $S$，可以**增量更新**（running sum），不需要显式 $n \times n$ attention matrix。

```python
q = self._phi(q)  # ELU+1
k = self._phi(k)
attn_num = q @ (k.transpose(-2, -1) @ v)           # φ(q) · S
attn_den = q @ k.sum(dim=2, keepdim=True).transpose(-2, -1)  # φ(q) · z
out = attn_num / (attn_den + 1e-6)
```

**复杂度**：$O(n \cdot d^2)$ vs full 的 $O(n^2 \cdot d)$。当 $n \gg d$ 时（长序列），linear 远快。

**劣势**：ELU+1 核对 softmax 的近似质量不如精确计算，长序列上信息会衰减——所以不能全用 linear，需要 full attention "定期"聚合信息。

#### 为什么 75/25

```python
n_linear = int(n_layers * linear_ratio)  # 75% 的层
for i in range(n_layers):
    attn_type = "linear" if i < n_linear else "full"
```

- **前 75% 层用 linear**：大部分层省显存 + 长上下文快
- **后 25% 层用 full**：保留关键层的 full attention，确保信息聚合质量
- 这个配比是 Qwen3.6 的**经验值**——在效率和质量的权衡中找到的甜点

!!! insight "见解：hybrid attention 是"用 25% 的代价保住质量""
    全用 linear attention 会损失质量（近似不够好），全用 full attention 会爆显存。
    75/25 的 hybrid 是"用 25% 的 full attention 代价保住大部分质量"——
    不是"linear 好还是 full 好"的二选一，而是**按层混合**。
    这和 Bonsai 的整体哲学一致：不追求单一极端，而是在约束下找最优配比。

## 3. 低 bit 切换：prepare_from_float

### 3.1 一键切换机制

所有 Linear 都能换成 `HadamardTernaryLinear`。`MiniBonsai(low_bit=True)` 构造时自动用低 bit 版：

```python
if low_bit:
    self.qkv = HadamardTernaryLinear(dim, 3 * dim, **lb_kwargs)
    self.o = HadamardTernaryLinear(dim, dim, **lb_kwargs)
else:
    self.qkv = nn.Linear(dim, 3 * dim)
    self.o = nn.Linear(dim, dim)
```

### 3.2 从 FP16 烧出低 bit 版

`prepare_from_float` 从一个训练好的 FP16 模型**离线量化**出低 bit 版：

```python
float_model = MiniBonsai(low_bit=False)           # FP16 版
# ... 训练 float_model ...

low_bit_model = MiniBonsai(low_bit=True, hybrid_attention=True)
low_bit_model.prepare_from_float(float_model)     # 烧出低 bit 版
```

内部流程：

```python
def prepare_from_float(self, float_model):
    self.embed.weight.data.copy_(float_model.embed.weight.data)  # embedding 直接拷
    self.norm_f.weight.data.copy_(float_model.norm_f.weight.data)  # norm 直接拷
    for lb, fb in zip(self.blocks, float_model.blocks):
        lb.prepare_from_float(fb)  # 每个 block 烧低 bit
    self.lm_head.prepare_from_float(float_model.lm_head)  # lm_head 烧低 bit
```

**关键**：

- **embedding 和 norm 不量化**——它们不是权重投影，保持 FP16
- **每个 Linear 烧低 bit**——`HadamardTernaryLinear.prepare_from_float` 做 Hadamard 旋转 + 三值化
- **结构保持一致**——低 bit 模型和 float 模型的层数、维度、attention 类型完全相同

### 3.3 前向时的低 bit 计算

低 bit 模型前向时，每个 `HadamardTernaryLinear` 做：

$$
y = (x \cdot H) \cdot (\text{trit} \cdot \text{scale})^\top \cdot H^\top
$$

1. 旋转 activation：$x' = x \cdot H$
2. 低 bit matmul：$y' = x' \cdot (\text{trit} \cdot \text{scale})^\top$
3. 反旋转：$y = y' \cdot H^\top$

详见 [Hadamard 旋转](../operators/hadamard.md) 和 [Ternary quantization](../operators/ternary.md)。

## 4. generate：带 KV cache 的自回归生成

### 4.1 流程

```python
def generate(self, tokens, n_new=10):
    kv_caches = [KVCache() for _ in self.blocks]  # 每层一个 cache
    logits = self.forward(tokens, kv_caches=kv_caches)  # prefill
    next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [next_tok]
    for _ in range(n_new - 1):
        logits = self.forward(next_tok, kv_caches=kv_caches)  # 每步只输入 1 个 token
        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(next_tok)
    return torch.cat([tokens] + generated, dim=1)
```

### 4.2 两阶段

1. **Prefill**：输入全部 prompt，算所有 token 的 attention，填充 KV cache
2. **Decode**：每步只输入 1 个新 token，用 KV cache 避免重算历史

### 4.3 使用示例

```python
from bonsai import MiniBonsai

model = MiniBonsai(
    vocab_size=1024, dim=256, n_layers=4,
    low_bit=True,
    hybrid_attention=True,
    linear_ratio=0.75,
)
model.prepare_from_float(float_model)

tokens = torch.randint(0, 1024, (1, 10))
generated = model.generate(tokens, n_new=50)  # (1, 60)
```

## 5. API 速查

```python
from bonsai import (
    RMSNorm, RoPE, KVCache, SwiGLUMLP,
    SimpleAttention, LinearAttention, SimpleBlock, MiniBonsai,
)

# 组件
norm = RMSNorm(dim=256)
rope = RoPE(head_dim=64, max_seq_len=2048)
cache = KVCache()
mlp = SwiGLUMLP(dim=256, intermediate=768, low_bit=True)

# attention
full_attn = SimpleAttention(dim=256, n_heads=8, low_bit=True, use_rope=True)
linear_attn = LinearAttention(dim=256, n_heads=8, low_bit=True)

# block
block = SimpleBlock(dim=256, n_heads=8, intermediate=768, attention_type="linear")

# 完整模型
model = MiniBonsai(low_bit=True, hybrid_attention=True, linear_ratio=0.75)
model.prepare_from_float(float_model)
generated = model.generate(tokens, n_new=50)
```

## 6. 进一步阅读

- **[Hadamard 旋转](../operators/hadamard.md)**：`HadamardTernaryLinear` 的旋转部分
- **[Ternary quantization](../operators/ternary.md)**：`HadamardTernaryLinear` 的三值化部分
- **[Why not collapse](../philosophy/why-not-collapse.md)**：hybrid attention 在"不塌陷"中的作用
- **`bonsai/block.py`**：完整实现