# Hadamard 旋转：正确性条件，不是优化

> **主线**：旋转不减小总误差，而是**改变误差的分布方向**——让误差不对齐任何敏感方向。
> 这个反直觉的洞察是 Bonsai 不塌陷的第一根支柱。
> 它不是"锦上添花的优化"，是**离了它就输出乱码的等价变换**。

---

## 1. 数学基础

### 1.1 Hadamard 矩阵的定义

Hadamard 矩阵 $H$ 是 $\pm 1$ 的正交矩阵（未归一化），满足：

$$H \cdot H^\top = n \cdot I$$

递归构造（Sylvester 构造）：

$$H_1 = [1], \quad H_{2n} = \begin{bmatrix} H_n & H_n \\ H_n & -H_n \end{bmatrix}$$

归一化后 $\hat{H} = H / \sqrt{n}$ 是真正的正交矩阵：$\hat{H} \cdot \hat{H}^\top = I$。

### 1.2 三个关键性质

**性质 1：正交性（保范数）**

$$\|W'\|_F = \|H \cdot W \cdot H^\top\|_F = \|W\|_F$$

旋转不改变权重的 Frobenius 范数——不丢信息，也不减小总误差。

**性质 2：各向同性（打散 outlier）**

$H$ 的关键性质：把任意方向的 outlier 能量**均匀打散**到所有方向。
旋转后的 $W'$ 接近球形高斯分布——各向同性，没有突出的"主方向"。

**性质 3：整数元素（无浮点误差）**

$H$ 的元素是 $\pm 1$，旋转可用加减法实现，无浮点误差。
这在低 bit 语境下特别重要——旋转本身不引入量化噪声。

### 1.3 随机化 Hadamard

单纯的 Walsh-Hadamard 有固定结构，旋转后可能留下 artifact（周期性 pattern）。
QuIP# 用 $H \cdot D$（$D$ 是随机对角符号矩阵）打散结构：

$$H_{\text{random}} = H \cdot \text{diag}(\pm 1, \pm 1, \ldots, \pm 1)$$

既保持正交性又避免周期性 pattern。Bonsai 的离线旋转即此类。

---

## 2. 为什么旋转能防止塌陷

### 2.1 塌陷的根因回顾

传统量化塌陷的原因：权重的**各向异性**。

- 少数方向（大奇异值方向）承载大部分信息，权重幅度大（$w = 5.3$）
- 多数方向接近零（$w = 0.01$）

三值化把大权重压扁（$5.3 \to +1$，丢失幅度），给小权重注入噪声（$0.01 \to +1$，凭空注入）。
**误差集中在敏感方向**——而下游任务最在乎这些方向。

### 2.2 旋转后的三值化

旋转把 $W$ 变成 $W' = H \cdot W \cdot H^\top$，$W'$ 接近球形高斯——**各向同性**。

三值化 $W'$ 的误差变成**各向均匀的小噪声**，而不是"沿主方向的系统性偏差"。

### 2.3 下游只在乎投影

下游误差的数学表达：

$$\text{误差}_{\text{下游}} = \sum_i \epsilon_i \cdot \langle \mathbf{e}_i, \mathbf{x}_{\text{activation}} \rangle$$

- $\epsilon_i$：第 $i$ 个方向的误差幅度
- $\mathbf{e}_i$：第 $i$ 个方向的单位向量
- $\mathbf{x}_{\text{activation}}$：activation 向量

**关键洞察**：下游不在乎误差的**总大小**，只在乎误差在 activation 方向的**投影**。

| 情况 | $\epsilon_i$ 分布 | $\langle \mathbf{e}_i, \mathbf{x} \rangle$ | 下游误差 |
|---|---|---|---|
| 不旋转 | 集中在敏感方向 | 大（敏感方向对齐 activation） | **大** |
| 旋转后 | 均匀小 | 小（无论 $\mathbf{x}$ 朝哪） | **小** |

---

## 3. 反直觉：增大权重误差但减小下游误差

### 3.1 实验 01 的核心发现

| 度量 | 无旋转 | 有旋转 | 变化 |
|---|---|---|---|
| 权重 Frobenius 误差 $\|W - \tilde{W}\|_F$ | 0.86 | 1.22 | **↑ 增大 42%** |
| 下游输出误差 $\|x \cdot W^\top - x \cdot \tilde{W}^\top\|$ | 0.86 | 0.52 | **↓ 减小 40%** |

### 3.2 为什么反直觉

!!! insight "旋转不减小总误差，但减小下游误差"
    旋转**不减小总误差**（正交变换保范数）：
    $$\|W - \tilde{W}\|_F = \|H(W - \tilde{W})H^\top\|_F$$
    实际上，因为三值化的阈值效应，旋转后的总误差甚至**更大**（1.22 > 0.86）。

    但下游误差**更小**（0.52 < 0.86）。因为旋转让误差**不对齐任何特定方向**——
    无论 activation 朝哪，误差投影都小。

    **传统度量（权重误差）误导了我们。** 真正重要的是"误差碰巧对齐了 activation 吗"，
    旋转让答案变成"几乎不对齐"。

### 3.3 这为什么是"正确性条件"

MODEL-FORMATS.md 明确警告：

> Older builds could load Q2_0 while missing those transforms and **produce gibberish**.

GGUF 文件里存的是**旋转后的三值权重** + rotation metadata。
Runtime 必须对 activation 应用匹配变换才能算对。

这不是"有旋转更好，没旋转差一点"——是"**没旋转就输出乱码**"。

---

## 4. 完整生命周期

### 4.1 离线（烧权重）

$$W' = H_{\text{blk}} \cdot W \cdot H_{\text{blk}}^\top \quad \text{（旋转到好量化的基）}$$
$$\text{trit}, \text{scale} = \text{ternarize}(W') \quad \text{（在旋转基下三值化）}$$

存储：trit (int8) + scale (float16) + $H$ (metadata，只需随机种子)

### 4.2 在线（forward）

$$x' = x \cdot H_{\text{blk}} \quad \text{（旋转 activation 到匹配基）}$$
$$y' = \text{ternary\_matmul}(x', \text{trit}, \text{scale}) \quad \text{（低 bit 乘，不解压）}$$
$$y = y' \cdot H_{\text{blk}}^\top \quad \text{（反旋转回原基）}$$

### 4.3 数学等价性

$$y = x \cdot W^\top \approx (x \cdot H) \cdot (H \cdot W \cdot H^\top)^\top \cdot (H^\top)^\top$$

逐项配对：$(x \cdot H)$ 旋转 activation，$(H \cdot W \cdot H^\top)^\top$ 旋转权重，$(H^\top)^\top$ 反旋转输出。

### 4.4 零额外成本

- **不花额外存储 bit**：$H$ 是 $\pm 1$ 矩阵，只需存随机种子（几个 byte）
- **不花额外权重带宽**：权重以三值形式存储，旋转不改变体积
- **activation 旋转开销小**：$H$ 是 $\pm 1$，旋转只需加减法

---

## 5. 分块旋转

### 5.1 为什么分块

完整的 $W' = H \cdot W \cdot H^\top$ 对大矩阵开销高（$O(n^3)$）。
实际做法是 **block-diagonal**：把 $W$ 切成 $\text{block\_size} \times \text{block\_size}$ 的小块，每块独立旋转。

这正是 Bonsai 模型卡片说的 "blockwise orthogonal Hadamard rotation"。

### 5.2 代码

```python
from bonsai import BlockHadamard

rot = BlockHadamard(block_size=128, randomized=True)
W_rot = rot.rotate_weight(W)        # 离线
x_rot = rot.rotate_activation(x)    # 在线
y = rot.rotate_output(y_rot)        # 在线
```

---

## 6. 和 QuIP# 的关系

QuIP# (Chee et al.) 是 Hadamard-domain quantization 的理论源头。
Bonsai 从 QuIP# 借鉴了"旋转到好量化的基"这个思想，但有关键区别：

| | QuIP# | Bonsai |
|---|---|---|
| 旋转 | 推理时做 | **折叠进存储**（离线做一次） |
| 量化 | 2-bit codebook | **三值化** $\{−1,0,+1\}$ |
| 额外 bit | 需要 codebook | **零额外 bit** |
| 训练 | PTQ | **QAT** |

Bonsai 的贡献：把旋转从"有开销的优化"变成"零开销的正确性条件"，
并配合三值化 + QAT 做到端到端。

---

## 7. 实验验证

`experiments/01_hadamard_magic.py` 的核心输出：

```
无旋转: 权重误差 0.86, 下游误差 0.86
有旋转: 权重误差 1.22 ↑, 下游误差 0.52 ↓
```

**读法**：

1. 权重误差增大（1.22 > 0.86）——旋转不减小总误差
2. 下游误差减小（0.52 < 0.86）——旋转让误差不对齐 activation
3. 这就是"旋转改变误差方向"的实验证据

---

## 进一步阅读

- [为什么 1.7 bit 不崩](../philosophy/why-not-collapse.md) — 旋转是三根支柱之一
- [三值化与二值化](ternary.md) — 旋转后在好基上做三值化
- `bonsai/hadamard.py` — 实现
- `experiments/01_hadamard_magic.py` — 实验