# Hadamard 旋转：正确性条件，不是优化

> **核心洞察**：旋转不是"锦上添花的优化"，是**离了它就输出乱码的等价变换**。
> 它不减小总误差，而是**改变误差的分布方向**——让误差不对齐任何敏感方向。

## 数学原理

### Hadamard 矩阵

Hadamard 矩阵 $H$ 是 $\pm 1$ 的正交矩阵（未归一化），满足：

$$H \cdot H^\top = n \cdot I$$

递归构造（Sylvester 构造）：

$$H_1 = [1], \quad H_{2n} = \begin{bmatrix} H_n & H_n \\ H_n & -H_n \end{bmatrix}$$

归一化后 $\hat{H} = H / \sqrt{n}$ 是真正的正交矩阵：$\hat{H} \cdot \hat{H}^\top = I$。

### 旋转的三个关键性质

1. **正交性**：$\|W'\|_F = \|H \cdot W \cdot H^\top\|_F = \|W\|_F$
   — Frobenius 范数不变，不丢信息

2. **各向同性**：把任意方向的 outlier 能量均匀打散到所有方向
   — 旋转后的 $W'$ 接近球形高斯分布

3. **整数元素**：$H$ 的元素是 $\pm 1$，旋转可用加减法实现，无浮点误差

### 随机化 Hadamard

单纯的 Walsh-Hadamard 有固定结构，旋转后可能留下 artifact。
QuIP# 用 $H \cdot D$（$D$ 是随机对角符号矩阵）打散结构：

$$H_{\text{random}} = H \cdot \text{diag}(\pm 1, \pm 1, \ldots, \pm 1)$$

既保持正交性又避免周期性 pattern。Bonsai 的离线旋转即此类。

## 完整生命周期

### 离线（烧权重）

$$W' = H_{\text{blk}} \cdot W \cdot H_{\text{blk}}^\top \quad \text{（旋转到好量化的基）}$$
$$\text{trit}, \text{scale} = \text{ternarize}(W') \quad \text{（在旋转基下三值化）}$$

存储：trit (int8) + scale (float16) + $H$ (metadata)

### 在线（forward）

$$x' = x \cdot H_{\text{blk}} \quad \text{（旋转 activation 到匹配基）}$$
$$y' = \text{ternary\_matmul}(x', \text{trit}, \text{scale}) \quad \text{（低 bit 乘，不解压）}$$
$$y = y' \cdot H_{\text{blk}}^\top \quad \text{（反旋转回原基）}$$

### 数学等价性

$$y = x \cdot W^\top \approx (x \cdot H) \cdot (H \cdot W \cdot H^\top)^\top \cdot (H^\top)^\top$$

逐项配对：$(x \cdot H)$ 旋转 activation，$(H \cdot W \cdot H^\top)^\top$ 旋转权重，$(H^\top)^\top$ 反旋转输出。

## 分块旋转

完整的 $W' = H \cdot W \cdot H^\top$ 对大矩阵开销高。实际做法是 **block-diagonal**：
把 $W$ 切成 $\text{block\_size} \times \text{block\_size}$ 的小块，每块独立旋转。

这正是 Bonsai 模型卡片说的 "blockwise orthogonal Hadamard rotation"。

```python
from bonsai import BlockHadamard

rot = BlockHadamard(block_size=128, randomized=True)
W_rot = rot.rotate_weight(W)        # 离线
x_rot = rot.rotate_activation(x)    # 在线
y = rot.rotate_output(y_rot)        # 在线
```

## 零额外成本

旋转折叠进存储权重（离线做一次），运行时只需对 activation 做匹配变换。

- **不花额外存储 bit**：$H$ 是 $\pm 1$ 矩阵，只需存随机种子（几个 byte）
- **不花额外权重带宽**：权重以三值形式存储，旋转不改变体积
- **activation 旋转开销小**：$H$ 是 $\pm 1$，旋转只需加减法

## 反直觉：增大权重误差但减小下游误差

实验 01 的核心发现：

| 度量 | 无旋转 | 有旋转 |
|---|---|---|
| 权重 Frobenius 误差 $\|W - \tilde{W}\|_F$ | 0.86 | **1.22** ↑ |
| 下游输出误差 $\|x \cdot W^\top - x \cdot \tilde{W}^\top\|$ | 0.86 | **0.52** ↓ |

!!! insight "为什么"
    旋转**不减小总误差**（正交变换保范数），而是**改变误差的分布方向**。

    下游误差 = $\sum_i \epsilon_i \cdot \langle \mathbf{e}_i, \mathbf{x}_{\text{activation}} \rangle$
    — 只在乎误差在 activation 方向的**投影**。

    旋转让误差均匀分布到所有方向，无论 activation 朝哪，投影都小。
    不旋转时，误差集中在权重的主方向（大奇异值方向），恰好对齐 activation → 投影大。

## 正确性条件，不是优化

MODEL-FORMATS.md 明确警告：

> Older builds could load Q2_0 while missing those transforms and **produce gibberish**.

GGUF 文件里存的是**旋转后的三值权重** + rotation metadata。
Runtime 必须对 activation 应用匹配变换才能算对。

这不是"有旋转更好，没旋转差一点"——是"没旋转就输出乱码"。

## API

::: bonsai.BlockHadamard
    selection:
      members:
        - rotate_weight
        - rotate_activation
        - rotate_output

---

详见 `bonsai/hadamard.py` 和 `experiments/01_hadamard_magic.py`。