# 实验解读指南

> 6 个实验各自展现 Bonsai 的一个关键面。这里讲清每个实验"看什么、怎么读"。

## 实验 01：Hadamard magic

**展现**：旋转**增大权重误差**但**减小下游误差**——QuIP# 的核心洞察。

```
无旋转: 权重误差 0.86, 下游误差 0.86
有旋转: 权重误差 1.22 ↑, 下游误差 0.52 ↓
```

**读法**：旋转不减小总误差（正交变换保范数），而是**改变误差的分布方向**。
下游只在乎误差在 activation 方向的投影——旋转让误差均匀分布，投影变小。

这是"为什么 Hadamard 旋转是正确性条件，不是优化"的实验证据。

## 实验 02：ternary vs binary

**展现**：端到端 MiniBonsai 前向对比——ternary vs binary vs float。

**读法**：

- ternary 一致优于 binary（多一个 0 值带来稀疏 + 精度）
- 纯 PTQ 下 +Hadamard 不一定更准（正交变换保奇异值，不改变 ternarize 总误差）
- Bonsai 的 98% 能力保留来自 QAT + 系统配合，不是单纯 PTQ + 旋转

## 实验 03：Intelligence Density

**展现**：Bonsai 的设计哲学度量。

$$D = \frac{-\log_2(1 - \text{score}/100)}{\text{size\_GB}}$$

**读法**：

- FP16 的 85 分在 54 GB 里密度很低（0.051）——装不下 = 0 分
- Bonsai 1-bit 的 76 分在 3.9 GB 里密度最高（0.513）——端侧最优
- "装不下 = 0 分"是硬约束，不是"性能差一点"

## 实验 04：QAT the real gain

**展现**：**QAT 才是 Bonsai 98% 能力的来源**。

```
PTQ（训练后三值化）: 损失 +0.90
QAT（训练时三值化）: 损失 +0.32
```

**读法**：

- PTQ 的权重是在 FP16 下训练的，没"学会"在 $\{−1, 0, +1\}$ 上工作
- QAT 用 STE 让梯度流过三值化，权重**主动适应**低 bit 约束
- 旋转 + QAT 是**乘法关系**：旋转准备好"好量化的基"，QAT 让权重学会在这个基上表达信息

## 实验 05：DSpark block-denoising

**展现**：DSpark 的 diffusion × speculative decoding 交叉创新。

```
code-like: 自回归接受率 0.97, block 0.96, block 快 4.2x
chat-like: 自回归接受率 0.28, block 0.29, block 快 4.1x
```

**读法**：

1. **block drafter 快 ~4x**（1 次前向 vs k 次）——这是 DSpark 的速度来源
2. **code-like 接受率远高于 chat-like**——DSpark 实测 code/math 2x、chat 收益少
3. **验证无损**：accepted token 和普通生成不可区分

**workload-dependent** 现象可见：加速依赖可预测性，不是免费的。

## 实验 06：XNOR-popcount + packing

**展现**：1-bit matmul 的位运算等价 + 两种 packing 格式。

```
Part 1: XNOR-popcount vs 浮点 matmul 误差 = 0（精确等价）
Part 2: 三值化 XNOR+mask 误差 = 0（需额外 mask 处理 0）
Part 3: PTQ1_0 1.725 bpw vs PQ2_0 2.125 bpw
Part 4: 理论算力比 256x（AVX2 XNOR vs FP16 FMA）
Part 5: packing roundtrip 误差 = 0
```

**读法**：

1. XNOR-popcount 是 1-bit matmul 的**精确等价**——不需要乘法器
2. 三值化的 0 需要 mask——这是 PTQ1_0/PQ2_0 复杂度的来源
3. PTQ1_0 更省空间但解码复杂，PQ2_0 更大但解码简单 → **架构敏感**
4. 1-bit 的硬件优势：位运算吞吐 >> 浮点乘加吞吐

## 实验的算子来源

所有实验的算子都从 `bonsai/` 包导入：

| 实验 | 导入来源 |
|---|---|
| 01-04 | `bonsai.hadamard`, `bonsai.quantize`, `bonsai.ops`, `bonsai.block` |
| 05 | `bonsai.dspark` |
| 06 | `bonsai.kernel` |

实验是**展示层**，包是**算子库**。这符合"收集完整模型实现"的设计——
`bonsai/` 是一个完整的模型库条目，实验只是调用包里的算子做可视化。