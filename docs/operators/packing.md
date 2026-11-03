# PTQ1_0 / PQ2_0 packing —— 三值化权重的两种打包格式

> **主线**：三值化权重 $\{-1, 0, +1\}$ 理论上只需 $\log_2 3 \approx 1.585$ bit，但硬件不能处理"1.585 个 bit"——
> 必须打包成整数字节。**怎么打包**是一个信息论 + 硬件工程的交叉问题：
> PTQ1_0 用 base-3 编码逼近信息论下界（1.6 bpw），PQ2_0 用 2-bit slot 换取解码速度（2.0 bpw）。
> 两者差 0.4 bpw，在 27B 模型上差 ~2 GB——但**哪个更快取决于硬件**，这不是实现差异，是架构差异。
> 本文从信息论下界出发，推导两种格式的编码/解码数学，分析它们的硬件行为，最后用实验验证 roundtrip 精度。

## 1. 为什么需要 packing

### 1.1 三值化的信息论下界

三值化权重 $w \in \{-1, 0, +1\}$ 有 3 个可能值，携带信息量：

$$
I(w) = \log_2 3 \approx 1.585 \text{ bit}
$$

这是**每个权重的理论最少 bit 数**。任何无损编码的平均 bit 数不能低于这个值（信源编码定理）。

但硬件以 **byte（8 bit）** 为最小寻址单位，不能存"1.585 个 bit"。必须把多个 trit 打包进整数字节。

### 1.2 打包的核心矛盾

打包格式面对一个根本性矛盾：

| 目标 | 方向 | 代价 |
|---|---|---|
| **省空间** | 逼近 1.585 bpw | 编码/解码复杂（非对齐访问） |
| **解码快** | 对齐到 2 bit | 浪费 slot 3（0.415 bit/trit 浪费） |

PTQ1_0 选前者，PQ2_0 选后者。**没有 universally 最优的 packing**——最优是 workload × 硬件的函数。

### 1.3 group scale 的额外开销

两种格式都配 group scale：每 128 个权重共享 1 个 FP16 scale（16 bit）。

$$
\text{scale\_bpw} = \frac{16}{128} = 0.125 \text{ bpw}
$$

这是**固定开销**，与 packing 格式无关。下文 bpw 都已包含这 0.125。

## 2. PTQ1_0：dense trit packing

### 2.1 信息论基础：为什么是 5 trit → 8 bit

PTQ1_0 的核心问题是：**多少个 trit 能塞进 1 个 byte？**

$n$ 个 trit 携带 $3^n$ 种状态，1 byte 有 $2^8 = 256$ 种状态。需要：

$$
3^n \leq 256
$$

取对数：

$$
n \leq \frac{8}{\log_2 3} = \frac{8}{1.585} \approx 5.05
$$

所以 $n = 5$：**5 个 trit 打包进 8 bit**，利用率：

$$
\eta = \frac{\log_2(3^5)}{8} = \frac{5 \times 1.585}{8} = \frac{7.925}{8} = 99.1\%
$$

99.1% 的字节利用率——几乎无损地逼近信息论下界。浪费的只有 $256 - 243 = 13$ 个编码点（5.1%）。

### 2.2 编码：base-3 整数

把 5 个 trit $\{-1, 0, +1\}$ 先映射到 $\{0, 1, 2\}$（加 1），再编码成 base-3 整数：

$$
\text{packed} = t_0 \cdot 3^4 + t_1 \cdot 3^3 + t_2 \cdot 3^2 + t_3 \cdot 3^1 + t_4 \cdot 3^0
$$

$$
= t_0 \cdot 81 + t_1 \cdot 27 + t_2 \cdot 9 + t_3 \cdot 3 + t_4
$$

其中 $t_i \in \{0, 1, 2\}$，$\text{packed} \in [0, 242]$，存进 1 个 uint8。

```python
t = (trits + 1).to(torch.int32)  # {-1,0,+1} → {0,1,2}
packed = t[:, 0] * 81 + t[:, 1] * 27 + t[:, 2] * 9 + t[:, 3] * 3 + t[:, 4]
```

### 2.3 解码：5 次除法 + 取模

解码是编码的逆——反复除以 3 取余数：

$$
t_4 = \text{packed} \bmod 3, \quad \text{packed} \leftarrow \lfloor \text{packed} / 3 \rfloor
$$

重复 5 次取出 $t_4, t_3, t_2, t_1, t_0$（逆序）。

```python
t4 = p % 3; p = p // 3
t3 = p % 3; p = p // 3
t2 = p % 3; p = p // 3
t1 = p % 3; p = p // 3
t0 = p
```

**5 次整数除法 + 5 次取模**——这是 PTQ1_0 的解码代价。在 SIMD 上除法/取模比位运算慢 3-10x。

### 2.4 密度

$$
\text{bpw}_{\text{PTQ1\_0}} = \frac{8}{5} + 0.125 = 1.6 + 0.125 = \mathbf{1.725 \text{ bpw}}
$$

vs 信息论下界 1.585 bpw——只多 0.115 bpw（7.3% 开销），几乎最优。

## 3. PQ2_0：2-bit slot packing

### 3.1 设计哲学：对齐换速度

PQ2_0 不追求信息论效率，而是**对齐到 2 bit**，让解码只用位运算（shift + mask）。

每个 trit $\{-1, 0, +1\}$ 映射到 $\{0, 1, 2\}$，存进 2 bit slot。slot 3（即 `0b11`）unused——浪费 25% 的 slot 空间，但换来了对齐友好。

### 3.2 编码：4 trit → 8 bit

4 个 trit 各占 2 bit，拼进 1 byte：

$$
\text{packed} = t_0 \,|\, (t_1 \ll 2) \,|\, (t_2 \ll 4) \,|\, (t_3 \ll 6)
$$

```python
t = (trits + 1).to(torch.int32)  # {0,1,2}
packed = t[:, 0] | (t[:, 1] << 2) | (t[:, 2] << 4) | (t[:, 3] << 6)
```

### 3.3 解码：shift + mask

每个 trit 用 mask `0x3` 取出，只需 shift + AND：

$$
t_0 = \text{packed} \,\&\, \texttt{0x3}, \quad
t_1 = (\text{packed} \gg 2) \,\&\, \texttt{0x3}, \quad \ldots
$$

```python
t0 = p & 0x3
t1 = (p >> 2) & 0x3
t2 = (p >> 4) & 0x3
t3 = (p >> 6) & 0x3
```

**4 次 shift + 4 次 AND**——纯位运算，无除法。在所有硬件上都是单周期。

### 3.4 密度

$$
\text{bpw}_{\text{PQ2\_0}} = 2 + 0.125 = \mathbf{2.125 \text{ bpw}}
$$

vs PTQ1_0 的 1.725 bpw——多 0.4 bpw（23% 开销）。在 27B 模型上：

$$
\Delta\text{size} = 27 \times 10^9 \times \frac{0.4}{8} \approx 1.35 \text{ GB}
$$

PQ2_0 比 PTQ1_0 多占 ~1.35 GB。

## 4. 密度对比与信息论视角

### 4.1 三种格式的完整对比

| 格式 | 数据 bpw | + scale | 总 bpw | 利用率 | 解码 |
|---|---|---|---|---|---|
| 1-bit (binary) | 1.0 | 0.125 | **1.125** | N/A | XNOR（最快） |
| **PTQ1_0** | 1.6 | 0.125 | **1.725** | 99.1% | 除法 + 取模（最紧） |
| **PQ2_0** | 2.0 | 0.125 | **2.125** | 79.2% | shift + mask（最简） |
| 理论下界 | 1.585 | 0.125 | **1.710** | 100% | 不可能 |

### 4.2 信息论效率

$$
\eta_{\text{PTQ1\_0}} = \frac{1.585}{1.6} = 99.1\%, \quad
\eta_{\text{PQ2\_0}} = \frac{1.585}{2.0} = 79.2\%
$$

PTQ1_0 几乎无损地逼近下界，PQ2_0 浪费了 20.8% 的空间。

!!! insight "见解：为什么 PTQ1_0 不用 6 trit → 10 bit？"
    $3^6 = 729 > 2^{10} = 1024$，6 trit 进 10 bit 的利用率是 $\frac{6 \times 1.585}{10} = 95.1\%$，
    **反而比 5→8 的 99.1% 低**。5→8 是 $3^n \le 2^8$ 的最优解。
    更大的块（如 11 trit → 18 bit，$\eta = 96.8\%$）利用率都不如 5→8。
    这是数论的一个非平凡事实：$(n, k) = (5, 8)$ 是 $\frac{n \log_2 3}{k}$ 的一个特别好近似。

## 5. 架构敏感性：为什么最优格式取决于硬件

两种格式的解码代价不同，在不同硬件上的相对速度也不同。**这不是实现差异，是架构差异**。

### 5.1 Ada / L4：PTQ1_0 更快

这些架构的 trit kernel 优化得好，除法可以用**查表 + 乘法近似**：

$$
p \div 3 \approx p \times \text{magic} \gg \text{shift}
$$

密集的 packing（1.6 bpw）让 **memory bandwidth 利用率高**——同样大小的 L2 cache 能装更多权重，减少 HBM 访问。在 memory-bound 场景（batch-1 推理），这是决定性优势。

### 5.2 H100 / Blackwell：PQ2_0 更快

这些架构的 **2-bit slot 对齐更好**。H100 的 Tensor Core 对 2-bit 边界友好，可以高效地做 2-bit gather。

shift + mask 的解码延迟比除法低 3-5x，**弥补了体积大的劣势**。在 compute-bound 场景（大 batch），解码速度比密度更重要。

### 5.3 Apple Silicon：1-bit XNOR 优势小

ARM NEON 的 SIMD 宽度是 128 bit，不如 x86 AVX2 的 256 bit 或 AVX-512 的 512 bit。

$$
\text{throughput}_{\text{XNOR}} \propto \text{SIMD width}
$$

位运算的硬件优势被较窄的 SIMD 抵消。在 M5 Max 上，1-bit 模型甚至**比 FP16 更慢**——这不是"还没优化"，是架构层面的根本性差异。

!!! insight "见解：Bonsai 提供两种格式的设计决策"
    Bonsai 不押注单一架构，而是**同时提供 PTQ1_0 和 PQ2_0**，让用户按硬件选。
    这本身就是一个设计哲学：**适配生态，而不是赌生态**。
    在 GPU 快速迭代、NPU/TPU 异构化的当下，押注单一 packing 格式是高风险的——
    今天的最优格式可能明天就不再是。提供两种格式是"用工程复杂度换架构鲁棒性"。

## 6. 实现细节

### 6.1 PTQ1_0 pack/unpack

```python
from bonsai import pack_trits_ptq10, unpack_trits_ptq10

trits = torch.randint(-1, 2, (40,))  # {−1, 0, +1}

packed = pack_trits_ptq10(trits)       # uint8, 8 bytes（40/5）
unpacked = unpack_trits_ptq10(packed)  # int8, 40 个
assert (trits == unpacked).all()       # 精确还原
```

**约束**：trit 数量必须是 5 的倍数（PTQ1_0 的块大小）。

### 6.2 PQ2_0 pack/unpack

```python
from bonsai import pack_trits_pq20, unpack_trits_pq20

packed = pack_trits_pq20(trits)        # uint8, 10 bytes（40/4）
unpacked = unpack_trits_pq20(packed)   # int8, 40 个
assert (trits == unpacked).all()       # 精确还原
```

**约束**：trit 数量必须是 4 的倍数（PQ2_0 的块大小）。

### 6.3 密度计算

```python
from bonsai import packing_bpw

packing_bpw()
# {
#     "1-bit (binary)": 1.125,
#     "PTQ1_0 (dense trit)": 1.725,
#     "PQ2_0 (2-bit slot)": 2.125,
# }
```

### 6.4 group scale 在 matmul 中的作用

两种格式都配 group scale。scale 是 FP16，在 matmul 时乘到结果上：

$$
y = \text{ternary\_matmul}(x, \text{trit}) \cdot \alpha_g
$$

其中 $\alpha_g$ 是第 $g$ 组的 scale（每 128 个权重共享一个）。这让三值化能表达权重的**幅度信息**——trit 只存符号，scale 存幅度。

## 7. 实验验证

`experiments/06_xnor_popcount_packing.py` Part 3 + Part 5：

### 7.1 密度验证

实验计算三种格式的 bpw，与理论值对比：

```
1-bit (binary):  1.125 bpw  ✓
PTQ1_0 (trit):   1.725 bpw  ✓
PQ2_0 (slot):    2.125 bpw  ✓
```

### 7.2 Roundtrip 精度

两种 packing 都做 pack → unpack roundtrip：

```
PTQ1_0 roundtrip 误差 = 0  ✓（精确还原）
PQ2_0  roundtrip 误差 = 0  ✓（精确还原）
```

**两种格式都是无损的**——pack/unpack 不引入任何误差。误差只来自三值化本身（$\{-1, 0, +1\}$ 近似 FP16 权重），不来自 packing。

!!! insight "见解：packing 是无损的，三值化才是有损的"
    这一点容易被忽略：packing 格式不影响模型精度，只影响**存储密度和解码速度**。
    模型精度完全由三值化 $\{-1, 0, +1\}$ + group scale 决定。
    所以选 PTQ1_0 还是 PQ2_0 是**纯工程决策**（速度 vs 空间），不是精度决策。

## 8. 进一步阅读

- **[XNOR-popcount kernel](xnor-popcount.md)**：1-bit matmul 的位运算等价——packing 的下游消费者
- **[Ternary quantization](ternary.md)**：三值化的赋值规则和 QAT 训练——packing 的上游生产者
- **[Intelligence Density](../philosophy/intelligence-density.md)**：为什么 0.4 bpw 的差距在端侧很重要
- **`bonsai/kernel.py`**：完整实现