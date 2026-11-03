# 实验解读指南 —— 6 个实验各自展现什么

> **主线**：6 个实验不是 6 个独立 demo——它们从不同角度验证 Bonsai 的**同一个核心主张**：
> "1.7 bit 还不崩"。每个实验揭示一个层次：实验 1 讲旋转为什么有效（QuIP# 洞察），
> 实验 2 讲端到端对比（ternary vs binary vs float），实验 3 讲设计哲学的度量（Intelligence Density），
> 实验 4 讲能力的真正来源（QAT 不是 PTQ），实验 5 讲推理加速（DSpark），
> 实验 6 讲硬件层（XNOR-popcount + packing）。
> 本文不只罗列实验结果，而是讲清**每个实验"看什么、怎么读、为什么这样设计"**——
> 把 6 个实验串成一条验证链。

## 1. 实验的设计逻辑

### 1.1 验证链

6 个实验构成一条从**算子**到**系统**的验证链：

```
算子层    实验 1（Hadamard）  →  旋转为什么有效
         实验 6（XNOR+pack）  →  硬件层怎么实现
模型层    实验 2（ternary vs binary）  →  端到端对比
         实验 4（QAT）  →  能力从哪来
哲学层    实验 3（Intelligence Density）  →  设计目标是什么
推理层    实验 5（DSpark）  →  怎么加速推理
```

### 1.2 每个实验的"反直觉发现"

| 实验 | 直觉预期 | 实际结果 | 反直觉发现 |
|---|---|---|---|
| 01 | 旋转减小误差 | 旋转**增大**权重误差 | 误差方向比大小重要 |
| 02 | +Hadamard 更准 | 纯 PTQ 下不稳定 | 旋转需要 QAT 配合 |
| 03 | 精度越高越好 | FP16 密度最低 | 装不下 = 0 分 |
| 04 | PTQ + 旋转就够 | QAT 才是来源 | 乘法关系 |
| 05 | DSpark universally 加速 | chat 收益少 | workload-dependent |
| 06 | 三值化纯 XNOR | 0 需要 mask | packing 架构敏感 |

## 2. 实验 01：Hadamard magic

### 2.1 展现什么

旋转**增大权重误差**但**减小下游误差**——QuIP# 的核心洞察。

```
无旋转: 权重误差 0.86, 下游误差 0.86
有旋转: 权重误差 1.22 ↑, 下游误差 0.52 ↓
```

### 2.2 怎么读

**关键**：旋转不减小总误差（正交变换保范数），而是**改变误差的分布方向**。

$$
\|W - \hat{W}\|_F = \|H(W - \hat{W})H^\top\|_F \quad \text{（正交变换保范数）}
$$

但下游误差是 $\|x(W - \hat{W})\|$，只在乎误差在 **activation 方向**的投影。旋转让误差**均匀分布**，投影变小：

$$
\|x(W - \hat{W})\| \downarrow \quad \text{即使} \quad \|W - \hat{W}\|_F \uparrow
$$

### 2.3 为什么这样设计

这个实验是"**为什么 Hadamard 旋转是正确性条件，不是优化**"的实验证据。

如果旋转只是"减小误差"，那任何减小误差的方法都行（比如更高 bit）。但旋转的独特价值是**改变误差方向**——这是其他方法做不到的。

### 2.4 算子来源

```python
from bonsai.hadamard import BlockHadamard
from bonsai.quantize import ternarize, dequantize
```

## 3. 实验 02：ternary vs binary

### 3.1 展现什么

端到端 MiniBonsai 前向对比——ternary vs binary vs float。

### 3.2 怎么读

- **ternary 一致优于 binary**：多一个 0 值带来稀疏 + 精度
- **纯 PTQ 下 +Hadamard 不一定更准**：正交变换保奇异值，不改变 ternarize 总误差
- **Bonsai 的 98% 能力保留来自 QAT + 系统配合**，不是单纯 PTQ + 旋转

### 3.3 为什么这样设计

这个实验是**反直觉的**——很多人以为"加了 Hadamard 旋转一定更准"，但纯 PTQ 下不一定。

旋转的收益需要 **QAT 配合**：旋转准备好"好量化的基"，QAT 让权重学会在这个基上表达信息。没有 QAT，权重没"学会"适应旋转后的基，旋转的收益无法兑现。

### 3.4 算子来源

```python
from bonsai import MiniBonsai  # bonsai.block
```

## 4. 实验 03：Intelligence Density

### 4.1 展现什么

Bonsai 的设计哲学度量。

$$
D = \frac{-\log_2(1 - \text{score}/100)}{\text{size\_GB}}
$$

### 4.2 怎么读

- **FP16 的 85 分在 54 GB 里密度很低**（0.051）——装不下 = 0 分
- **Bonsai 1-bit 的 76 分在 3.9 GB 里密度最高**（0.513）——端侧最优
- **"装不下 = 0 分"是硬约束**，不是"性能差一点"

### 4.3 为什么这样设计

这个实验回答"**为什么不用更高 bit**"——不是"精度不够好"，是"装不下"。

端侧 AI 的真目标是"每 GB 的可用智力"，不是绝对精度。FP16 的 85 分在 54 GB 里是浪费——如果设备只有 8 GB 内存，FP16 模型根本装不下，精度再高也是 0。

### 4.4 算子来源

```python
# 纯数学计算，不导入算子
```

## 5. 实验 04：QAT the real gain

### 5.1 展现什么

**QAT 才是 Bonsai 98% 能力的来源**。

```
PTQ（训练后三值化）: 损失 +0.90
QAT（训练时三值化）: 损失 +0.32
```

### 5.2 怎么读

- **PTQ 的权重是在 FP16 下训练的**，没"学会"在 $\{-1, 0, +1\}$ 上工作
- **QAT 用 STE 让梯度流过三值化**，权重**主动适应**低 bit 约束
- **旋转 + QAT 是乘法关系**：旋转准备好"好量化的基"，QAT 让权重学会在这个基上表达信息

### 5.3 为什么这样设计

这个实验是**最重要的反直觉发现**——很多人以为"Bonsai 的秘密是 Hadamard 旋转"，但**不是**。

旋转是**正确性条件**（不塌陷的必要条件），但**能力的来源是 QAT**。没有 QAT，旋转只能做到 PTQ 的水平（损失 +0.90）；有 QAT，才能做到 98% 保留（损失 +0.32）。

### 5.4 乘法关系的实验证据

$$
\text{PTQ 无旋转}: +0.90 \quad \text{PTQ + 旋转}: +0.85 \quad \text{QAT + 旋转}: +0.32
$$

- 旋转在 PTQ 下只改善 0.05（**加法关系**，微弱）
- QAT 改善 0.53（**主要贡献**）
- 旋转在 QAT 下进一步改善（**乘法关系**，旋转让 QAT 更有效）

### 5.5 算子来源

```python
from bonsai.quantize import ternarize  # STE 在 ternarize 里
from bonsai.ops import HadamardTernaryLinear
```

## 6. 实验 05：DSpark block-denoising

### 6.1 展现什么

DSpark 的 diffusion × speculative decoding 交叉创新。

```
code-like: 自回归接受率 0.97, block 0.96, block 快 4.2x
chat-like: 自回归接受率 0.28, block 0.29, block 快 4.1x
```

### 6.2 怎么读

1. **block drafter 快 ~4x**（1 次前向 vs $k=4$ 次）——这是 DSpark 的速度来源，与 workload 无关
2. **code-like 接受率远高于 chat-like**（0.96 vs 0.29）——**workload-dependent** 现象可见
3. **block 和自回归的接受率几乎相同**——block-denoising 的质量损失在本实验中很小
4. **验证无损**：accepted token 和普通生成不可区分

### 6.3 为什么这样设计

这个实验展现"**加速不是免费的**"——DSpark 的加速比 = 速度提升 × 接受率。

- code-like 的有效加速比 = 4.2x × 0.96 ≈ **4.0x**（几乎完美）
- chat-like 的有效加速比 = 4.1x × 0.29 ≈ **1.2x**（收益很小）

**workload-dependent** 现象可见：加速依赖可预测性，不是免费的。

### 6.4 算子来源

```python
from bonsai.dspark import (
    TargetModel, AutoregressiveDrafter, BlockDenoisingDrafter,
    verify, train_drafter, make_sequence,
)
```

## 7. 实验 06：XNOR-popcount + packing

### 7.1 展现什么

1-bit matmul 的位运算等价 + 两种 packing 格式。

```
Part 1: XNOR-popcount vs 浮点 matmul 误差 = 0（精确等价）
Part 2: 三值化 XNOR+mask 误差 = 0（需额外 mask 处理 0）
Part 3: PTQ1_0 1.725 bpw vs PQ2_0 2.125 bpw
Part 4: 理论算力比 256x（AVX2 XNOR vs FP16 FMA）
Part 5: packing roundtrip 误差 = 0
```

### 7.2 怎么读

1. **XNOR-popcount 是 1-bit matmul 的精确等价**——不需要乘法器
2. **三值化的 0 需要 mask**——这是 PTQ1_0/PQ2_0 复杂度的来源
3. **PTQ1_0 更省空间但解码复杂，PQ2_0 更大但解码简单** → **架构敏感**
4. **1-bit 的硬件优势**：位运算吞吐 >> 浮点乘加吞吐

### 7.3 为什么这样设计

这个实验讲清"**1-bit 模型在端侧快的根本原因**"——不是"权重少所以快"，是"位运算吞吐远高于浮点乘加"。

$$
\text{理论算力比} = \frac{\text{AVX2 XNOR 吞吐}}{\text{FP16 FMA 吞吐}} = 256\text{x}
$$

XNOR 一次处理 256 bit（AVX2），FP16 FMA 一次 1 个——256 倍的原始算力优势。

### 7.4 算子来源

```python
from bonsai.kernel import (
    xnor_popcount_matmul, ternary_matmul_via_xnor,
    packing_bpw, pack_trits_ptq10, unpack_trits_ptq10,
    pack_trits_pq20, unpack_trits_pq20,
)
```

## 8. 实验的算子来源汇总

所有实验的算子都从 `bonsai/` 包导入：

| 实验 | 导入来源 | 展现层次 |
|---|---|---|
| 01 | `bonsai.hadamard`, `bonsai.quantize` | 算子层（旋转） |
| 02 | `bonsai.block` | 模型层（端到端） |
| 03 | 纯数学 | 哲学层（度量） |
| 04 | `bonsai.quantize`, `bonsai.ops` | 模型层（QAT） |
| 05 | `bonsai.dspark` | 推理层（加速） |
| 06 | `bonsai.kernel` | 算子层（硬件） |

实验是**展示层**，包是**算子库**。这符合"收集完整模型实现"的设计——`bonsai/` 是一个完整的模型库条目，实验只是调用包里的算子做可视化。

## 9. 怎么跑

```bash
pip install -r requirements.txt
python experiments/01_hadamard_magic.py
python experiments/02_ternary_vs_binary.py
python experiments/03_intelligence_density.py
python experiments/04_qat_the_real_gain.py
python experiments/05_dspark_block_denoising.py
python experiments/06_xnor_popcount_packing.py
```

所有实验都是**自包含的**——不依赖外部数据或预训练模型，几秒内跑完。

## 10. 进一步阅读

- **[Hadamard 旋转](../operators/hadamard.md)**：实验 01 的算子
- **[Ternary quantization](../operators/ternary.md)**：实验 02、04 的算子
- **[Intelligence Density](../philosophy/intelligence-density.md)**：实验 03 的哲学
- **[Why not collapse](../philosophy/why-not-collapse.md)**：实验 04 的"乘法关系"
- **[DSpark](../operators/dspark.md)**：实验 05 的算子
- **[XNOR-popcount + packing](../operators/packing.md)**：实验 06 的算子