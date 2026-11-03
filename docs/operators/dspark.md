# DSpark block-denoising drafter

> **核心独创**：diffusion × speculative decoding 的交叉——
> 把 diffusion 的"整块去噪"嫁接到 token drafting。

## 传统 speculative decoding

小模型（drafter）逐个**自回归**起草 token → 大模型（target）逐个验证。

```
drafter: 草 t1 → 草 t2(t1) → 草 t3(t1,t2) → ...  （串行 k 次前向）
target:  验证 t1, t2, t3, ...                    （一次并行验证）
```

**瓶颈**：drafting 本身是串行的——草第 $i$ 个 token 要看前 $i-1$ 个。
加速比受限于 drafting 速度，不是验证速度。

## DSpark 的创新

drafter 是 **block-parallel** 的，一次并行草 $k$ 个 token。

```
drafter: 草 [t1, t2, t3, t4] 一次性  （1 次前向）
target:  验证 t1, t2, t3, t4         （一次并行验证）
```

用 **diffusion-flavored block-denoising objective** 训练 drafter——
给定 prefix 一次性预测整块 $k$ 个 token，而不是逐个自回归。

### 速度

草 $k$ 个 token 只要 **1 次前向**（vs 自回归的 $k$ 次）。

### 质量

可能比自回归差——每个 token 看不到同块后面的 token（没有 causal 依赖）。
但**验证保无损**：target 逐个检查，不匹配的截断，accepted token 和普通生成不可区分。

## 训练细节

### survival-probability-weighted distillation

训练时给"更可能被接受的位置"更高权重。

speculative decoding 的接受概率随位置递减（越后面越可能被拒），
所以训练时给前面的位置高权重，让 drafter 在"最可能被接受的位置"上更准。

### per-source-normalized hidden-state taps

从 target 的 5 个均匀间隔层 tap hidden state，做归一化后作为 drafter 输入。

这让 drafter 能"看到" target 的中间表示，不只是最终 logits——
信息更丰富，drafter 能更好地模仿 target 的行为。

### draft block size $k$ 的选择

$k$ 从 **measured verify-cost model** 选——先测验证成本，再反推最优 $k$。

验证成本 = target 验证 $k$ 个 token 的时间。
drafting 成本 = drafter 1 次前向的时间。
最优 $k$ = 让 (drafting + verify) 最小的 $k$。

## 反直觉的 workload × 硬件依赖

| 场景 | L40S (CUDA) | M5 Max (Metal) |
|---|---|---|
| ternary code/math | 1.8–2.4x | ~1.2x |
| ternary chat/reasoning | 较少 | **更慢** |
| 1-bit 任何场景 | 1.4–1.75x | **更慢** |

### code/math 收益最大

代码和数学的下一个 token 高度可预测（语法结构强），drafter 接受率高（~0.9）。
block-denoising 的"看不到同块后面"的劣势小——因为高可预测性意味着 $t_{i+1}$ 基本由 prefix 决定，不太依赖 $t_i$。

### chat/reasoning 收益少

自然语言多样性高，下一个 token 不确定性强，接受率低。
block-denoising 的劣势大——$t_{i+1}$ 强依赖 $t_i$，看不到就猜不准。

### Apple Silicon 上更慢

batch-1 的验证 pass 在 Metal 上不 amortize。
这不是"还没优化"，是**架构层面的根本性差异**——Metal 的 batch-1 kernel 启动开销高。

## 代价

- **跨请求 prompt-cache 复用被禁**：每个请求要 re-prefill 全部历史
- **单 slot**（`-np 1`）：不能并行处理多个请求

这是"用吞吐换延迟"的选择——适合交互式端侧场景，不适合高并发服务端。

## 上游化

DSpark 已 upstream 到 mainline llama.cpp（PR #25173，`draft-dspark`）。
标志着 Bonsai 的算法创新已被社区认可。

## 代码

```python
from bonsai.dspark import (
    TargetModel, AutoregressiveDrafter, BlockDenoisingDrafter,
    verify, train_drafter, make_sequence,
)

# 造数据
data = make_sequence("code", 500, vocab=100)

# 训练 drafter
target = TargetModel(vocab=100)
block_drafter = BlockDenoisingDrafter(vocab=100, block_size=4)
train_drafter(block_drafter, target, data)

# 起草 + 验证
drafted = block_drafter.draft(prefix, k=4)      # 1 次前向
accepted = verify(target, prefix, drafted)       # 无损验证
```

## 实验验证

`experiments/05_dspark_block_denoising.py`：

| 序列类型 | 自回归接受率 | block 接受率 | block 速度 |
|---|---|---|---|
| code-like | 0.97 | 0.96 | 4.2x 快 |
| chat-like | 0.28 | 0.29 | 4.1x 快 |

code-like 接受率远高于 chat-like——**workload-dependent** 现象可见。

---

详见 `bonsai/dspark.py`。