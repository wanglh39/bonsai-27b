# 为什么 1.7 bit 不崩

> **核心问题**：传统量化在 sub-4-bit 会"塌陷"（math 掉到 57 分），Bonsai 在 1.7 bit 还保留 98% 能力。
> 为什么？答案不是一个技术，而是**三个技术的系统配合**。

## 塌陷的原因：权重的各向异性

真实大模型的权重矩阵是**各向异性**的：少数方向（大奇异值方向）承载大部分信息，
多数方向接近零。这是训练自然产生的结构——模型把重要信息编码在特定方向上。

三值化 $\{−1, 0, +1\}$ 做了什么：

1. **大权重被压扁**：一个承载关键信息的权重 $w = 5.3$ 被量化成 $+1$，丢失了幅度信息
2. **小权重被注入噪声**：一个不重要的 $w = 0.01$ 也被量化成 $+1$，凭空注入了等量噪声

关键在于：误差**不是均匀分布**的。大权重方向（敏感方向）的误差最大，
而下游任务（attention、reasoning chain）恰恰最在乎这些方向。

$$\text{误差}_{\text{下游}} = \sum_i \epsilon_i \cdot \langle \mathbf{e}_i, \mathbf{x}_{\text{activation}} \rangle$$

其中 $\mathbf{e}_i$ 是误差方向，$\mathbf{x}_{\text{activation}}$ 是 activation。
如果 $\mathbf{e}_i$ 集中在敏感方向，$\langle \mathbf{e}_i, \mathbf{x} \rangle$ 就大 → 下游崩盘。

## 第一根支柱：Hadamard 旋转

### 旋转做了什么

$$W' = H \cdot W \cdot H^\top$$

其中 $H$ 是 Hadamard 矩阵（$\pm 1$ 的正交矩阵，$H \cdot H^\top = n \cdot I$）。

$H$ 的关键性质：**把任意方向的能量均匀打散到所有方向**。旋转后的 $W'$ 接近球形高斯分布——
各向同性，没有突出的"主方向"。

### 旋转后的三值化误差

三值化 $W'$ 的误差变成**各向均匀的小噪声**，而不是"沿主方向的系统性偏差"。
下游只在乎误差在 activation 方向的投影：

$$\text{误差}_{\text{下游}} = \sum_i \epsilon'_i \cdot \langle \mathbf{e}'_i, \mathbf{x} \rangle$$

旋转后 $\epsilon'_i$ 均匀小，无论 $\mathbf{x}$ 朝哪个方向，投影都小。

!!! warning "反直觉：旋转增大权重误差但减小下游误差"
    实验 01 验证：Hadamard 旋转**增大权重 Frobenius 误差**（0.86 → 1.22），
    但**减小下游输出误差**（0.86 → 0.52）。

    旋转不减小总误差（正交变换保范数），而是**改变误差的分布方向**。
    下游不在乎总误差大小，只在乎"误差碰巧对齐了 activation 吗"——
    旋转让误差不对齐任何特定方向。

### 零额外成本

旋转折叠进存储权重（离线做一次 $W' = H \cdot W \cdot H^\top$），
运行时只需对 activation 做匹配变换 $x' = x \cdot H$。

**不花额外存储 bit，不花额外权重带宽**。MODEL-FORMATS.md 明确警告：

> Bonsai 2 needs the Hadamard/sign-flip integration as well as format decoding:
> update old runtimes before using Q2_0. Older builds could load Q2_0 while missing
> those transforms and **produce gibberish**.

旋转不是"可选优化"，是**正确性的必要条件**——离了它就输出乱码。

## 第二根支柱：QAT（训练时三值化）

### PTQ vs QAT

| 方法 | 做法 | Bonsai 实测损失 |
|---|---|---|
| PTQ（训练后量化） | 训练完 FP16 → 再三值化 | +0.90 |
| **QAT（量化感知训练）** | **训练时就三值化前向** | **+0.32** |

PTQ 的权重是在 FP16 下训练的，没有"学会"在 $\{−1, 0, +1\}$ 上工作。
QAT 用 Straight-Through Estimator（STE）让梯度流过不可导的三值化函数，
权重**主动适应**低 bit 约束。

!!! insight "Bonsai 的 98% 能力保留，QAT 是主要来源"
    旋转把"好量化的基"准备好了，但权重还需要"学会"在这个基上用 $\{−1, 0, +1\}$ 表达信息。
    QAT 让权重学会这件事。两者是**乘法关系**，不是加法。

### STE 的原理

三值化 $w \mapsto \text{round}(w/\Delta) \in \{−1, 0, +1\}$ 是不可导的（阶梯函数）。
STE 的做法：前向用三值化，反向把梯度直接传给原始 $w$（假装三值化是恒等函数）。

$$\frac{\partial \tilde{w}}{\partial w} \approx 1 \quad \text{(STE)}$$

这让梯度能流回 $w$，权重在训练中逐步移动到"三值化后损失最小"的位置。

## 第三根支柱：Hybrid Attention

Qwen3.6 的 hybrid attention：75% 的层用 **linear attention**（$O(n)$ 复杂度），
25% 用 full self-attention（$O(n^2)$）。

- Linear attention 用核技巧 $\phi(q) \cdot \phi(k)^\top$ 近似 softmax，省显存 + 长上下文快
- Full attention 保留在关键层，确保信息聚合质量

这不直接帮助"不崩"，但它在**低 bit 约束下最大化有效上下文长度**——
同样是 Intelligence Density 的体现（每 GB 能处理多少 token）。

## 三根支柱的乘法关系

$$\text{保留能力} = \underbrace{\text{旋转}}_{\text{改变误差方向}} \times \underbrace{\text{QAT}}_{\text{权重适应}} \times \underbrace{\text{hybrid}}_{\text{上下文效率}}$$

- 去掉旋转：误差沿敏感方向集中 → 塌陷
- 去掉 QAT：权重没适应低 bit → 损失 +0.90
- 去掉 hybrid：长上下文显存爆炸 → 装不下

三者缺一不可。Bonsai 的贡献不是发明其中任何一个，而是**把它们端到端整合成一个无 escape hatch 的系统**。

## 实验验证

| 实验 | 展现 |
|---|---|
| `01_hadamard_magic` | 旋转增大权重误差但减小下游误差 |
| `02_ternary_vs_binary` | 端到端 MiniBonsai 前向对比 |
| `04_qat_the_real_gain` | QAT vs PTQ：+0.32 vs +0.90 |