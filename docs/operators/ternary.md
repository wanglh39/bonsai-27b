# 三值化与二值化

> **核心**：把权重从 FP16 压到 $\{−1, 0, +1\}$（三值）或 $\{−1, +1\}$（二值），
> 每 128 个权重共享一个 FP16 scale。幅度信息交给 scale，方向信息交给 trit/sign。

## 三值化 (Ternary)

### 表示

权重 $\tilde{w} \in \{−1, 0, +1\}$，每 $\text{group\_size}$ 个权重共享一个 FP16 scale $\alpha_g$。

一个 trit 承载 $\log_2 3 \approx 1.585$ bit 信息，
加 16-bit scale 摊到 128 个权重上 → 有效 ~1.71 bits/weight。

### 赋值规则（BitNet 1.58b 路线）

$$\alpha_g = \text{mean}(|w_g|) \quad \text{（group scale，组的平均幅度）}$$
$$\Delta_g = 0.7 \cdot \alpha_g \quad \text{（阈值，低于它判 0）}$$
$$\tilde{w} = \text{clip}\left(\text{round}\left(\frac{w}{\Delta_g}\right), -1, 1\right)$$

反量化：$\hat{w} = \tilde{w} \cdot \Delta_g$

### 为什么阈值是 0.7 * mean(|w|)

经验上让 **~一半权重为 0**（稀疏），非零权重保留符号和量级主信息。

0 的好处：

1. **稀疏 → matmul 可跳过**：0 位置的乘法直接跳过
2. **信息保留**：大权重保留符号（±1），小权重归零（本来就不重要）
3. **与 XNOR 交互**：0 需要 mask 处理，这是 PTQ1_0/PQ2_0 复杂度的来源

### 代码

```python
from bonsai import ternarize, dequantize

trit, scale = ternarize(W, group_size=128)
# trit: int8, 值 ∈ {−1, 0, +1}
# scale: float, 每个 group 一个

W_recovered = dequantize(trit, scale, group_size=128)
```

## 二值化 (Binary)

### 表示

权重 $\tilde{w} \in \{−1, +1\}$，**无 0**。更激进：幅度信息全交给 group scale。

$$\alpha_g = \text{mean}(|w_g|)$$
$$\tilde{w} = \text{sign}(w)$$

反量化：$\hat{w} = \tilde{w} \cdot \alpha_g$

有效 **1.125 bits/weight**（1 sign bit + 16-bit scale / 128）。

### 1-bit 的硬件优势

二值化权重的 matmul 可以用 **XNOR + popcount** 实现——纯位运算，不需要乘法器。

详见 [XNOR-popcount kernel](xnor-popcount.md)。

### 代价

去掉 0 意味着**所有权重都非零**——没有稀疏性可利用。
幅度信息全靠 group scale，精度比三值化低。

实测：1-bit Bonsai 保留 89.5% 能力，ternary 保留 98.2%。

## group_size = 128 的甜点

group 越小 → scale 越密 → 精度越高但 scale 开销越大。

| group_size | scale 开销 (bpw) | 精度 |
|---|---|---|
| 32 | 0.5 | 高 |
| **128** | **0.125** | **甜点** |
| 256 | 0.0625 | 低 |

128 让 scale 开销压到可忽略（0.125 bpw），同时精度够好。
Bonsai 模型卡片的 "Q1_0_g128" / "ternary g128" 即此。

## PTQ vs QAT

| 方法 | 做法 | 损失 |
|---|---|---|
| PTQ | 训练完 FP16 → 再三值化 | +0.90 |
| **QAT** | 训练时就三值化前向（STE） | **+0.32** |

QAT 用 Straight-Through Estimator 让梯度流过不可导的三值化：

$$\frac{\partial \tilde{w}}{\partial w} \approx 1 \quad \text{(STE)}$$

权重在训练中**主动适应**低 bit 约束，学会在 $\{−1, 0, +1\}$ 上表达信息。

详见 `experiments/04_qat_the_real_gain.py`。

## API

```python
from bonsai import ternarize, binarize, dequantize

# 三值化
trit, scale = ternarize(W, group_size=128, threshold_ratio=0.7)

# 二值化
sign, scale = binarize(W, group_size=128)

# 反量化
W_approx = dequantize(trit, scale, group_size=128)
```

---

详见 `bonsai/quantize.py`。