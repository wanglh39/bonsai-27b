# DSpark block-denoising drafter —— diffusion × speculative decoding

> **主线**：传统 speculative decoding 的瓶颈不在验证，而在 **drafting 本身是串行的**——
> 草第 $i$ 个 token 要看前 $i-1$ 个，$k$ 个 token 需要 $k$ 次前向。
> DSpark 的核心独创是把 diffusion 的"整块去噪"嫁接到 token drafting：
> drafter **一次并行草 $k$ 个 token**，只需 1 次前向。
> 但 block-parallel 草的每个 token 看不到同块后面的 token（无 causal 依赖），质量可能下降——
> DSpark 用三个训练技巧（survival-weighted distillation、hidden-state taps、cost-based block size）弥补，
> 并靠 target 的逐个验证**保无损**。
> 本文从传统 spec decoding 的瓶颈出发，推导 DSpark 的加速比公式，分析三个训练技巧的作用，
> 最后用 code-like vs chat-like 实验展现**加速是 workload-dependent 的**——不是免费的。

## 1. 传统 speculative decoding 的瓶颈

### 1.1 基本流程

Speculative decoding 用一个小模型（drafter）起草，大模型（target）验证：

```
drafter: 草 t₁ → 草 t₂(t₁) → 草 t₃(t₁,t₂) → ...  （串行 k 次前向）
target:  验证 t₁, t₂, t₃, ...                    （1 次并行验证）
```

- **drafter** 逐个**自回归**起草 $k$ 个 token，每个依赖前一个
- **target** 一次性并行验证所有 $k$ 个 token，接受匹配的前缀，从第一个不匹配处截断

### 1.2 加速比公式

设：

- $t_d$ = drafter 一次前向时间
- $t_v$ = target 验证 $k$ 个 token 的时间（1 次并行前向）
- $\alpha$ = 接受率（每个位置被接受的概率，假设独立）

传统 spec decoding 的总时间：

$$
T_{\text{trad}} = k \cdot t_d + t_v
$$

期望接受 token 数：$E[\text{accepted}] = \frac{1 - (1-\alpha)^k}{\alpha}$（几何分布前缀）。

加速比（vs 纯 target 自回归）：

$$
S_{\text{trad}} = \frac{E[\text{accepted}] \cdot t_v}{k \cdot t_d + t_v}
$$

### 1.3 瓶颈：drafting 串行

当 $k$ 增大时，**$k \cdot t_d$ 线性增长**，而 $t_v$ 近似常数（并行验证）。加速比的分子趋于 $1/\alpha$（常数），分母趋于 $k \cdot t_d$——**加速比随 $k$ 增大而下降**。

!!! insight "见解：瓶颈不在验证，在 drafting"
    传统 spec decoding 的优化方向是"让 drafter 更准"（提高 $\alpha$）或"让 drafter 更小"（降低 $t_d$）。
    但这些都治标不治本——**drafting 的串行性是结构性瓶颈**。
    DSpark 的创新是直接消除这个串行性：block-parallel drafting。

## 2. DSpark 的创新：block-parallel drafting

### 2.1 核心想法

drafter **不逐个草**，而是给定 prefix 一次性预测整块 $k$ 个 token：

```
drafter: 草 [t₁, t₂, t₃, t₄] 一次性  （1 次前向）
target:  验证 t₁, t₂, t₃, t₄         （1 次并行验证）
```

### 2.2 加速比改进

DSpark 的总时间：

$$
T_{\text{DSpark}} = t_d + t_v
$$

加速比：

$$
S_{\text{DSpark}} = \frac{E[\text{accepted}] \cdot t_v}{t_d + t_v}
$$

vs 传统的 $k \cdot t_d + t_v$——**drafting 成本从 $k \cdot t_d$ 降到 $t_d$**，$k$ 倍加速。

当 $t_d \ll t_v$（drafter 远小于 target）时：

$$
S_{\text{DSpark}} \approx E[\text{accepted}] \approx \frac{1 - (1-\alpha)^k}{\alpha}
$$

理论上接近 $1/\alpha$（$\alpha$ 高时接近 $k$）。

### 2.3 质量代价：无 causal 依赖

block-parallel 草的每个 token $t_i$ **看不到同块后面的 token $t_{i+1}, \ldots, t_k$**——它们是同时预测的，没有 causal 依赖。

这可能导致质量下降：如果 $t_{i+1}$ 强依赖 $t_i$（如自然语言），block drafter 在 $t_{i+1}$ 上的准确率低于自回归 drafter。

### 2.4 无损保证：target 验证

但**验证保无损**：target 逐个检查每个 drafted token 是否匹配自己的 argmax，从第一个不匹配处截断。

$$
\text{accepted} = \text{最长前缀使得 } t_i^{\text{draft}} = \text{target}.\text{argmax}(\text{prefix} + t_1^{\text{draft}} + \ldots + t_{i-1}^{\text{draft}})
$$

accepted token 和普通自回归生成**不可区分**——DSpark 的加速不牺牲精度，只可能"白草"（drafted token 被拒，浪费 drafter 前向）。

!!! insight "见解：DSpark 是"用 drafter 前向换 target 前向"的交易"
    DSpark 的本质是：用 1 次 drafter 前向（便宜）赌 $k$ 个 token，target 验证（贵）只在有匹配时才"赚到"。
    如果接受率高（$\alpha \approx 1$），1 次 drafter + 1 次 target 验证 ≈ $k$ 次 target 前向 → $k$ 倍加速。
    如果接受率低（$\alpha \approx 0$），1 次 drafter + 1 次 target 验证 ≈ 1 次 target 前向 → 没有加速，还浪费了 drafter 前向。
    **加速比完全由 $\alpha$ 决定**，而 $\alpha$ 由 workload 决定。

## 3. 训练细节：三个关键设计

block-parallel drafter 不能直接用"预测下一个 token"的交叉熵训练——它要**一次预测 $k$ 个 token**。DSpark 用三个技巧训练。

### 3.1 Block-denoising objective

给定 prefix，drafter 一次输出 $k$ 个 logits（对每个位置 $i \in [1, k]$ 预测 $\text{token}_{\text{prefix}+i}$）：

$$
\mathcal{L} = \sum_{i=1}^{k} w_i \cdot \text{CE}(\text{logits}_i, \text{target\_token}_i)
$$

其中 $w_i$ 是位置 $i$ 的权重（见下文 survival-weighted）。

这和 diffusion 的 block-denoising 类似——给定"被遮蔽的块"一次性预测整块，而不是逐个自回归。**这是"diffusion × speculative decoding"交叉的来源**。

### 3.2 Survival-probability-weighted distillation

speculative decoding 的接受概率随位置递减——越后面的 token 越可能被拒（因为前面的误差会累积）。

DSpark 给**更可能被接受的位置**（前面的位置）更高权重：

$$
w_i \propto \prod_{j=1}^{i-1} \alpha_j \quad \text{(survival probability)}
$$

其中 $\alpha_j$ 是位置 $j$ 的接受概率。这让 drafter 在"最可能被接受的位置"上更准——**优化的是有效 token 数，不是所有位置的均匀精度**。

### 3.3 Per-source-normalized hidden-state taps

drafter 不只看 token id，还从 target 的 **5 个均匀间隔层 tap hidden state**，做归一化后作为额外输入：

$$
\text{drafter\_input} = [\text{embed}(t), \, \text{norm}(h^{(l_1)}), \, \text{norm}(h^{(l_2)}), \, \ldots, \, \text{norm}(h^{(l_5)})]
$$

这让 drafter 能"看到" target 的中间表示，不只是最终 logits——信息更丰富，drafter 能更好地模仿 target 的行为。

**为什么是 5 层？** 太少（1-2 层）信息不够，太多（所有层）开销大且冗余。5 层是经验值——足够覆盖 target 的不同抽象层次。

### 3.4 Draft block size $k$ 的选择

$k$ 从 **measured verify-cost model** 选——先测验证成本，再反推最优 $k$。

$$
k^* = \arg\max_k \frac{E[\text{accepted}(k)] \cdot t_v}{t_d + t_v(k)}
$$

其中 $t_v(k)$ 是 target 验证 $k$ 个 token 的时间（随 $k$ 缓慢增长，因为并行）。

**这不是"越大越好"**——$k$ 太大时接受率下降（后面的 token 越难猜），且 $t_v(k)$ 增长。最优 $k$ 是接受率和验证成本的平衡点。

## 4. 反直觉的 workload × 硬件依赖

DSpark 的加速比**强 workload-dependent × 硬件-dependent**。这是它最反直觉的特性——不是" universally 加速"。

### 4.1 实测加速比

| 场景 | L40S (CUDA) | M5 Max (Metal) |
|---|---|---|
| ternary code/math | 1.8–2.4x | ~1.2x |
| ternary chat/reasoning | 较少 | **更慢** |
| 1-bit 任何场景 | 1.4–1.75x | **更慢** |

### 4.2 为什么 code/math 收益最大

代码和数学的下一个 token **高度可预测**（语法结构强、重复模式多），drafter 接受率高（$\alpha \approx 0.9$）。

block-denoising 的"看不到同块后面"的劣势小——因为高可预测性意味着 $t_{i+1}$ 基本由 prefix 决定，不太依赖 $t_i$：

$$
P(t_{i+1} \mid \text{prefix}, t_i) \approx P(t_{i+1} \mid \text{prefix})
$$

所以 block-parallel 草的准确率接近自回归草，加速比接近 $k$。

### 4.3 为什么 chat/reasoning 收益少

自然语言多样性高，下一个 token 不确定性强，接受率低（$\alpha \approx 0.3$）。

block-denoising 的劣势大——$t_{i+1}$ 强依赖 $t_i$（语义连贯性），看不到就猜不准：

$$
P(t_{i+1} \mid \text{prefix}, t_i) \gg P(t_{i+1} \mid \text{prefix})
$$

block-parallel 草的准确率远低于自回归草，接受率低，加速比小。

### 4.4 为什么 Apple Silicon 上更慢

batch-1 的验证 pass 在 Metal 上**不 amortize**——kernel 启动开销高，并行验证 $k$ 个 token 的成本不比 $k$ 次串行低多少。

这不是"还没优化"，是**架构层面的根本性差异**——Metal 的 batch-1 kernel 启动开销高，与 CUDA 的 sustained throughput 模型不同。

!!! insight "见解：DSpark 不是" universally 加速"，是"特定场景加速""
    DSpark 的加速依赖 **workload 可预测性 × 硬件 batch 效率** 两个条件同时满足。
    这意味着 DSpark 的适用场景是**明确的**：code/math 生成 + CUDA 硬件。
    在其他场景（chat/reasoning + Metal）上，DSpark 可能**不加速甚至减速**。
    这不是缺陷——这是对"speculative decoding 能 universally 加速"这个假设的修正。

## 5. 代价

DSpark 的加速不是免费的，有两个结构性代价：

### 5.1 跨请求 prompt-cache 复用被禁

DSpark 的 drafter 需要从 target tap hidden state，这要求 **target 和 drafter 在同一进程**，且 target 的中间状态可用。

这禁用了跨请求的 prompt-cache 复用——每个请求要 **re-prefill 全部历史**，不能复用之前请求的 KV cache。

### 5.2 单 slot（`-np 1`）

不能并行处理多个请求——drafter 和 target 必须独占 GPU。

### 5.3 适用场景

这是"**用吞吐换延迟**"的选择：

- **适合**：交互式端侧场景（单用户、低延迟优先）
- **不适合**：高并发服务端（多用户、吞吐优先）

## 6. 上游化：社区认可

DSpark 已 upstream 到 mainline llama.cpp（PR #25173，`draft-dspark`）。

这标志着 Bonsai 的算法创新**已被社区认可**——不是只存在于论文的 prototype，而是进入了主流推理框架。

## 7. 代码

```python
from bonsai.dspark import (
    TargetModel, AutoregressiveDrafter, BlockDenoisingDrafter,
    verify, train_drafter, make_sequence,
)

# 造数据：code-like（高可预测）vs chat-like（低可预测）
data = make_sequence("code", 500, vocab=100)

# 训练 drafter
target = TargetModel(vocab=100)
block_drafter = BlockDenoisingDrafter(vocab=100, block_size=4)
train_drafter(block_drafter, target, data)

# 起草 + 验证
drafted = block_drafter.draft(prefix, k=4)      # 1 次前向（vs 自回归 4 次）
accepted = verify(target, prefix, drafted)       # 无损验证
```

### 7.1 BlockDenoisingDrafter 的结构

```python
class BlockDenoisingDrafter:
    embed: Embedding(vocab, dim)
    rnn: GRU(dim, dim)
    head: Linear(dim, vocab * block_size)  # 一次输出 k 个 logits
```

`head` 的输出维度是 `vocab * block_size`——一次前向输出 $k$ 个位置的 logits，reshape 成 $(k, \text{vocab})$ 后 argmax。

### 7.2 vs AutoregressiveDrafter

```python
class AutoregressiveDrafter:
    embed: Embedding(vocab, dim)
    rnn: GRU(dim, dim)
    head: Linear(dim, vocab)  # 只输出 1 个 logit
```

自回归 drafter 的 `head` 只输出 1 个 logit，`draft(prefix, k)` 要循环 $k$ 次，每次把新 token 拼到 prefix 上再前向。

## 8. 实验验证

`experiments/05_dspark_block_denoising.py`：

### 8.1 接受率对比

| 序列类型 | 自回归接受率 | block 接受率 | block 速度 |
|---|---|---|---|
| code-like | 0.97 | 0.96 | 4.2x 快 |
| chat-like | 0.28 | 0.29 | 4.1x 快 |

### 8.2 怎么读

1. **block drafter 快 ~4x**（1 次前向 vs $k=4$ 次）——这是 DSpark 的速度来源，与 workload 无关
2. **code-like 接受率远高于 chat-like**（0.96 vs 0.29）——**workload-dependent** 现象可见
3. **block 和自回归的接受率几乎相同**（0.96 vs 0.97, 0.29 vs 0.28）——block-denoising 的质量损失在本实验中很小
4. **验证无损**：accepted token 和普通生成不可区分

### 8.3 code-like 为什么接受率高

`make_sequence("code")` 生成重复 pattern + 少量 noise：

```python
pattern = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
seq = pattern.repeat(length // 10 + 1)  # 强重复结构
```

下一个 token 高度可预测（$\alpha \approx 0.97$），block drafter 也能准确预测。

### 8.4 chat-like 为什么接受率低

`make_sequence("chat")` 生成随机序列 + 少量 pattern：

```python
seq = torch.randint(0, vocab, (length,))  # 几乎全随机
```

下一个 token 不确定性强（$\alpha \approx 0.29$），drafter 很难猜对——**这是 workload 的本质差异，不是 drafter 的缺陷**。

!!! insight "见解：加速比 = 速度 × 接受率"
    code-like 的有效加速比 = 4.2x × 0.96 ≈ **4.0x**（几乎完美）
    chat-like 的有效加速比 = 4.1x × 0.29 ≈ **1.2x**（收益很小）
    DSpark 的实际收益是"速度提升 × 接受率"——速度提升恒定（~4x），接受率由 workload 决定。
    所以 DSpark 的适用场景是**高可预测 workload**（code/math），不是所有场景。

## 9. 进一步阅读

- **[Innovation lineage](../philosophy/innovation-lineage.md)**：DSpark 在 Bonsai 谱系中的位置——独创 vs 借鉴
- **[Intelligence Density](../philosophy/intelligence-density.md)**：为什么端侧延迟比吞吐重要
- **[Transformer 架构](../architecture/transformer.md)**：target 模型的结构
- **`bonsai/dspark.py`**：完整实现