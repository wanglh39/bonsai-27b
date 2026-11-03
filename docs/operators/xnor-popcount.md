# XNOR-popcount kernel

> **核心**：1-bit 权重 $\{−1, +1\}$ 的 matmul **不需要乘法器**，只需 XNOR + popcount（位运算）。
> 这是 1-bit 模型在端侧快的**根本原因**。

## 数学等价

### 从乘法到位运算

对于 $a, b \in \{−1, +1\}$：

$$a \cdot b = \begin{cases} +1 & \text{if } a = b \\ -1 & \text{if } a \neq b \end{cases}$$

把 $\{−1, +1\}$ 映射到 $\{0, 1\}$：$a \mapsto \frac{a+1}{2}$。
乘法等价于 **XNOR**（相同为 1，不同为 0）：

$$a \cdot b = 2 \cdot \text{XNOR}(a_{\text{bit}}, b_{\text{bit}}) - 1$$

### 点积

$$\mathbf{a} \cdot \mathbf{b} = \sum_k a_k b_k = 2 \cdot \text{popcount}(\text{XNOR}(\mathbf{a}, \mathbf{b})) - n$$

其中 popcount 是数 1 的个数，XNOR 是按位同或。

### matmul

$$Y = A \cdot B = 2 \cdot \text{popcount}(\text{XNOR}(A_{\text{bits}}, B_{\text{bits}})) - K$$

整个 matmul 只用 XNOR + popcount + 加减法，**没有一次乘法**。

## 硬件优势

### 理论 throughput

| 操作 | 一次处理 | 吞吐 |
|---|---|---|
| FP16 FMA | 1 个 FP16 乘加 | 1 op/cycle |
| AVX2 XNOR | 256 bit | **256 op/cycle** |
| AVX-512 XNOR | 512 bit | **512 op/cycle** |

理论算力比：**256x ~ 512x**（取决于 SIMD 宽度）。

### 实际因素

理论 256x 不会完全兑现，因为：

1. **scale 乘法**：group scale 是 FP16，需要一次乘法 per group
2. **activation 量化**：activation 不是 1-bit，需要先量化
3. **memory bandwidth**：1-bit 权重省带宽，但 activation 还是 FP16

但即使打折扣，1-bit 在端侧（batch=1，memory-bound）的实测加速仍很显著。

## 代码

```python
from bonsai import xnor_popcount_matmul

a = torch.sign(torch.randn(32, 128))  # {−1, +1}
b = torch.sign(torch.randn(128, 16))  # {−1, +1}

# 浮点 matmul
float_result = a.float() @ b.float()

# XNOR-popcount matmul（数学等价）
xnor_result = xnor_popcount_matmul(a, b)

assert torch.allclose(float_result, xnor_result)  # 误差 = 0
```

## 三值化的挑战

三值化 $\{−1, 0, +1\}$ 有 0，**不能纯 XNOR**——0 位置的贡献是 0，不是 ±1。

### 解法：mask + XNOR

拆成：

- $\text{nonzero\_mask} = (b \neq 0)$
- $b_{\text{sign}} = \text{sign}(b) \in \{−1, +1\}$

只在非零位置做 XNOR-popcount：

$$\mathbf{a} \cdot \mathbf{b} = \text{XNOR-popcount}(\mathbf{a}_{\text{nonzero}}, \text{sign}(\mathbf{b}_{\text{nonzero}}))$$

### 额外开销

vs 纯 1-bit：

1. **mask 计算 + gather**：非零位置不连续 → 不规则内存访问
2. **load imbalance**：每列非零数不同
3. **存 magnitude**：需要额外 bit 标记 0/非零

这是 PTQ1_0 / PQ2_0 packing 复杂度的根本来源。详见 [packing](packing.md)。

```python
from bonsai import ternary_matmul_via_xnor

a = torch.sign(torch.randn(32, 128))      # {−1, +1}
b = torch.randint(-1, 2, (128, 16))       # {−1, 0, +1}

result = ternary_matmul_via_xnor(a, b)    # XNOR + mask
```

## Apple Silicon 的劣势

ARM NEON 的 XNOR 吞吐不如 x86 AVX2/AVX-512。

- x86：AVX2 256 bit，AVX-512 512 bit
- ARM：NEON 128 bit（AArch64）

这解释了为什么 Bonsai 1-bit 在 Apple Silicon 上优势小——
位运算的硬件优势被 SIMD 宽度差距抵消了。

## 实验验证

`experiments/06_xnor_popcount_packing.py` Part 1-2 验证：

- XNOR-popcount 和浮点 matmul 误差 = 0（精确等价）
- 三值化 XNOR+mask 和浮点 matmul 误差 = 0

---

详见 `bonsai/kernel.py`。