# 三值化与二值化

> **主线**：把权重从 FP16 压到 $\{−1, 0, +1\}$（三值）或 $\{−1, +1\}$（二值），
> 幅度信息交给 group scale，方向信息交给 trit/sign。
> 关键不是"压得多狠"，而是"**怎么让权重学会在低 bit 上工作**"——这就是 QAT。

---

## 1. 三值化 (Ternary)

### 1.1 表示

权重 $\tilde{w} \in \{−1, 0, +1\}$，每 $\text{group\_size}$ 个权重共享一个 FP16 scale $\alpha_g$。

一个 trit 承载 $\log_2 3 \approx 1.585$ bit 信息，
加 16-bit scale 摊到 128 个权重上 → 有效 ~1.71 bits/weight。

### 1.2 赋值规则（BitNet 1.58b 路线）

$$\alpha_g = \text{mean}(|w_g|) \quad \text{（group scale，组的平均幅度）}$$
$$\Delta_g = 0.7 \cdot \alpha_g \quad \text{（阈值，低于它判 0）}$$
$$\tilde{w} = \text{clip}\left(\text{round}\left(\frac{w}{\Delta_g}\right), -1, 1\right)$$

反量化：$\hat{w} = \tilde{w} \cdot \Delta_g$

### 1.3 为什么阈值是 0.7 * mean(|w|)

这个 0.7 不是随便选的。它让 **~一半权重为 0**（稀疏），非零权重保留符号和量级主信息。

0 的三个好处：

1. **稀疏 → matmul 可跳过**：0 位置的乘法直接跳过，省计算
2. **信息保留**：大权重保留符号（±1），小权重归零（本来就不重要）
3. **与 XNOR 交互**：0 需要 mask 处理，这是 PTQ1_0/PQ2_0 复杂度的来源

为什么是 0.7 而不是 0.5 或 0.8？

- 0.5：太多权重为 0，信息丢失严重
- 0.8：太少权重为 0，稀疏性不够
- **0.7**：经验甜点，~50% 稀疏 + 信息保留好

这是 BitNet 1.58b 论文的经验值，Bonsai 直接继承。

### 1.4 代码

```python
from bonsai import ternarize, dequantize

trit, scale = ternarize(W, group_size=128)
# trit: int8, 值 ∈ {−1, 0, +1}
# scale: float, 每个 group 一个

W_recovered = dequantize(trit, scale, group_size=128)
```

---

## 2. 二值化 (Binary)

### 2.1 表示

权重 $\tilde{w} \in \{−1, +1\}$，**无 0**。更激进：幅度信息全交给 group scale。

$$\alpha_g = \text{mean}(|w_g|)$$
$$\tilde{w} = \text{sign}(w)$$

反量化：$\hat{w} = \tilde{w} \cdot \alpha_g$

有效 **1.125 bits/weight**（1 sign bit + 16-bit scale / 128）。

### 2.2 1-bit 的硬件优势

二值化权重的 matmul 可以用 **XNOR + popcount** 实现——纯位运算，不需要乘法器。

理论 throughput：AVX2 XNOR 一次 256 bit，vs FP16 FMA 一次 1 个 → **256x 算力比**。

详见 [XNOR-popcount kernel](xnor-popcount.md)。

### 2.3 代价

去掉 0 意味着**所有权重都非零**——没有稀疏性可利用。
幅度信息全靠 group scale，精度比三值化低。

实测：1-bit Bonsai 保留 89.5% 能力，ternary 保留 98.2%。

但 1-bit 的 **Intelligence Density 更高**（0.513 vs 0.437）——
因为体积更小（3.9 GB vs 5.9 GB），在端侧预算下装得下才是王道。

---

## 3. group_size = 128 的甜点

### 3.1 权衡

group 越小 → scale 越密 → 精度越高但 scale 开销越大。

| group_size | scale 开销 (bpw) | 精度 | 总 bpw |
|---|---|---|---|
| 32 | 0.5 | 高 | 2.085 |
| **128** | **0.125** | **甜点** | **1.71** |
| 256 | 0.0625 | 低 | 1.648 |

### 3.2 为什么 128

128 让 scale 开销压到可忽略（0.125 bpw），同时精度够好。

- 太小（32）：scale 占 0.5 bpw，总 2.085 bpw——和 2-bit 量化差不多了，没省
- 太大（256）：scale 省，但一个 scale 管 256 个权重，精度下降
- **128**：scale 只占 0.125 bpw，精度够好——甜点

Bonsai 模型卡片的 "Q1_0_g128" / "ternary g128" 即此。

---

## 4. PTQ vs QAT：核心差异

### 4.1 两种方法

| 方法 | 做法 | 权重适应 | Bonsai 实测损失 |
|---|---|---|---|
| PTQ（训练后量化） | 训练完 FP16 → 再三值化 | 否 | +0.90 |
| **QAT（量化感知训练）** | **训练时就三值化前向** | **是** | **+0.32** |

### 4.2 PTQ 的问题

PTQ 的权重是在 FP16 下训练的，**没有"学会"在 $\{−1, 0, +1\}$ 上工作**。
三值化是训练后强加的，权重的分布和三值化的粒度不匹配。

具体来说：

- FP16 训练出的权重分布是连续的，有很多"不大不小"的值（比如 $w = 0.5$）
- 三值化把 $w = 0.5$ 判成 0（低于阈值 $0.7 \cdot \alpha$）——但这个权重可能很重要
- 或者把 $w = 0.8$ 判成 +1——但它的真实幅度是 0.8，不是 1.0

PTQ 没有机会调整这些"边界 case"——训练已经结束了。

### 4.3 QAT 的做法

QAT 在训练时前向传播就用三值化权重：

```python
# QAT 前向
w_real = layer.weight                    # FP32 参数
w_ternary = ternarize(w_real)            # 三值化（前向用这个）
output = x @ w_ternary.float()           # 前向用三值化权重

# QAT 反向（STE）
loss.backward()                          # 梯度直接流给 w_real
optimizer.step()                        # w_real 根据梯度更新
```

权重在训练中**主动适应**低 bit 约束——
"边界 case"的权重会被梯度推到"三值化后损失最小"的位置。

### 4.4 STE：让梯度流过不可导的三值化

三值化函数 $w \mapsto \text{round}(w/\Delta) \in \{−1, 0, +1\}$ 是**不可导**的（阶梯函数）。
反向传播需要梯度，但阶梯函数的导数几乎处处为 0。

**Straight-Through Estimator (STE)** 的做法：

$$\frac{\partial \tilde{w}}{\partial w} \approx 1 \quad \text{(STE)}$$

前向用三值化（$\tilde{w} = \text{ternarize}(w)$），反向把梯度**直接传给**原始 $w$（假装三值化是恒等函数）。

这不"正确"（三值化的真实梯度不是 1），但**有效**——
它让权重 $w$ 能收到梯度，在训练中逐步移动到"三值化后损失最小"的位置。

### 4.5 QAT 和旋转的乘法关系

!!! insight "旋转和 QAT 不是加法，是乘法"
    - 旋转把"好量化的基"准备好了——权重在旋转后的基上更各向同性
    - QAT 让权重**学会**在这个基上用 $\{−1, 0, +1\}$ 表达信息

    两者是**乘法关系**：
    - 只有旋转没 QAT：基好了但权重没适应 → PTQ 损失 +0.90
    - 只有 QAT 没旋转：权重适应了但基不好 → 误差仍沿敏感方向集中
    - **两者都有**：基好 + 权重适应 → QAT 损失 +0.32

---

## 5. QAT 的代价

QAT 不是免费的：

1. **训练成本高**：需要在三值化约束下训练，前向/反向都多一步
2. **需要完整训练数据**：不能像 PTQ 那样"拿现成模型压一下"
3. **训练稳定性**：STE 的梯度是近似的，训练初期可能不稳定

但这些都是**一次性成本**（训练时付），换来的是**永久收益**（推理时一直享受低 bit + 高精度）。
对于发布给大量用户使用的模型，这个权衡是值得的。

---

## 6. 实验验证

`experiments/04_qat_the_real_gain.py` 的核心输出：

```
PTQ（训练后三值化）: 损失 +0.90
QAT（训练时三值化）: 损失 +0.32
```

**读法**：

1. PTQ 损失 +0.90——权重没适应低 bit，三值化是强加的
2. QAT 损失 +0.32——权重学会了在 $\{−1, 0, +1\}$ 上工作
3. **QAT 是 Bonsai 98% 能力保留的主要来源**

---

## 进一步阅读

- [为什么 1.7 bit 不崩](../philosophy/why-not-collapse.md) — QAT 是三根支柱之一
- [Hadamard 旋转](hadamard.md) — 旋转 + QAT 的乘法关系
- [XNOR-popcount kernel](xnor-popcount.md) — 二值化的硬件优势
- `bonsai/quantize.py` — 实现