# XNOR-popcount kernel

> **主线**：1-bit 权重 $\{−1, +1\}$ 的 matmul **不需要乘法器**，只需 XNOR + popcount（位运算）。
> 这是 1-bit 模型在端侧快的**根本原因**——不是"快一点"，是"换了一种计算方式"。
> 但三值化的 0 打破了这个等价，需要额外 mask——这是 PTQ1_0/PQ2_0 复杂度的来源。

---

## 1. 数学等价：从乘法到位运算

### 1.1 核心观察

对于 $a, b \in \{−1, +1\}$：

$$a \cdot b = \begin{cases} +1 & \text{if } a = b \\ -1 & \text{if } a \neq b \end{cases}$$

乘法的结果只取决于"两个数是否相同"——这正是 **XNOR**（同或）的逻辑！

### 1.2 从 $\{−1, +1\}$ 到 $\{0, 1\}$

把 $\{−1, +1\}$ 映射到 $\{0, 1\}$：$a \mapsto \frac{a+1}{2}$。

在这个映射下，乘法等价于 XNOR：

$$a \cdot b = 2 \cdot \text{XNOR}(a_{\text{bit}}, b_{\text{bit}}) - 1$$

| $a$ | $b$ | $a \cdot b$ | $a_{\text{bit}}$ | $b_{\text{bit}}$ | XNOR | $2 \cdot \text{XNOR} - 1$ |
|---|---|---|---|---|---|---|
| +1 | +1 | +1 | 1 | 1 | 1 | +1 ✓ |
| +1 | −1 | −1 | 1 | 0 | 0 | −1 ✓ |
| −1 | +1 | −1 | 0 | 1 | 0 | −1 ✓ |
| −1 | −1 | +1 | 0 | 0 | 1 | +1 ✓ |

### 1.3 点积

$$\mathbf{a} \cdot \mathbf{b} = \sum_k a_k b_k = 2 \cdot \text{popcount}(\text{XNOR}(\mathbf{a}, \mathbf{b})) - n$$

其中 popcount 是数 1 的个数，XNOR 是按位同或。

**整个点积只用 XNOR + popcount + 加减法，没有一次乘法。**

### 1.4 matmul

$$Y = A \cdot B = 2 \cdot \text{popcount}(\text{XNOR}(A_{\text{bits}}, B_{\text{bits}})) - K$$

---

## 2. 硬件优势

### 2.1 理论 throughput

| 操作 | 一次处理 | 吞吐 |
|---|---|---|
| FP16 FMA | 1 个 FP16 乘加 | 1 op/cycle |
| AVX2 XNOR | 256 bit | **256 op/cycle** |
| AVX-512 XNOR | 512 bit | **512 op/cycle** |

理论算力比：**256x ~ 512x**（取决于 SIMD 宽度）。

### 2.2 为什么差距这么大

FP16 FMA 是"标量"操作——一条指令算一次乘加。
XNOR 是"SIMD"操作——一条指令同时处理 256/512 个 bit。

1-bit 权重每个只占 1 bit，256 个权重塞进一个 256-bit 寄存器，一条 XNOR 指令全部处理。
FP16 权重每个占 16 bit，一个 256-bit 寄存器只能塞 16 个，一条 FMA 指令处理 16 个。

### 2.3 实际因素

理论 256x 不会完全兑现，因为：

1. **scale 乘法**：group scale 是 FP16，需要一次乘法 per group
2. **activation 量化**：activation 不是 1-bit，需要先量化
3. **memory bandwidth**：1-bit 权重省带宽，但 activation 还是 FP16

但即使打折扣，1-bit 在端侧（batch=1，memory-bound）的实测加速仍很显著。

---

## 3. 代码验证

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

实验 06 Part 1 验证：**误差 = 0**，XNOR-popcount 是 1-bit matmul 的精确等价。

---

## 4. 三值化的挑战

### 4.1 0 打破了等价

三值化 $\{−1, 0, +1\}$ 有 0，**不能纯 XNOR**——
0 位置的贡献是 0，不是 ±1，XNOR 无法表达"这个位置不算"。

### 4.2 解法：mask + XNOR

拆成：

- $\text{nonzero\_mask} = (b \neq 0)$ — 哪些位置有值
- $b_{\text{sign}} = \text{sign}(b) \in \{−1, +1\}$ — 非零位置的符号

只在非零位置做 XNOR-popcount：

$$\mathbf{a} \cdot \mathbf{b} = \text{XNOR-popcount}(\mathbf{a}_{\text{nonzero}}, \text{sign}(\mathbf{b}_{\text{nonzero}}))$$

### 4.3 额外开销

vs 纯 1-bit：

| 开销 | 原因 |
|---|---|
| mask 计算 + gather | 非零位置不连续 → 不规则内存访问 |
| load imbalance | 每列非零数不同 |
| 存 magnitude | 需要额外 bit 标记 0/非零 |

这是 PTQ1_0 / PQ2_0 packing 复杂度的**根本来源**。
详见 [packing](packing.md)。

### 4.4 代码

```python
from bonsai import ternary_matmul_via_xnor

a = torch.sign(torch.randn(32, 128))      # {−1, +1}
b = torch.randint(-1, 2, (128, 16))       # {−1, 0, +1}

result = ternary_matmul_via_xnor(a, b)    # XNOR + mask
```

实验 06 Part 2 验证：**误差 = 0**，三值化 XNOR+mask 也是精确等价。

---

## 5. Apple Silicon 的劣势

### 5.1 SIMD 宽度差距

| 架构 | SIMD 宽度 | XNOR 吞吐 |
|---|---|---|
| x86 AVX2 | 256 bit | 256 op/cycle |
| x86 AVX-512 | 512 bit | 512 op/cycle |
| ARM NEON (AArch64) | 128 bit | 128 op/cycle |

ARM NEON 的 XNOR 吞吐只有 AVX2 的一半、AVX-512 的四分之一。

### 5.2 这解释了什么

Bonsai 1-bit 在 Apple Silicon 上优势小——
位运算的硬件优势被 SIMD 宽度差距抵消了。

这也是 DSpark 在 M5 Max 上反而更慢的原因之一——
1-bit 的 kernel 优势在 ARM 上不显著，但 DSpark 的 batch-1 验证开销在 Metal 上很高。

---

## 6. 实验验证

`experiments/06_xnor_popcount_packing.py` Part 1-2：

```
Part 1: XNOR-popcount vs 浮点 matmul 误差 = 0（精确等价）
Part 2: 三值化 XNOR+mask 误差 = 0（需额外 mask 处理 0）
Part 4: 理论算力比 256x（AVX2 XNOR vs FP16 FMA）
```

**读法**：

1. XNOR-popcount 是 1-bit matmul 的**精确等价**——不需要乘法器
2. 三值化的 0 需要 mask——这是 PTQ1_0/PQ2_0 复杂度的来源
3. 1-bit 的硬件优势：位运算吞吐 >> 浮点乘加吞吐

---

## 进一步阅读

- [PTQ1_0/PQ2_0 packing](packing.md) — 三值化的两种打包格式
- [三值化与二值化](ternary.md) — 为什么三值化有 0
- `bonsai/kernel.py` — 实现