# PTQ1_0 / PQ2_0 packing

> **核心**：三值化权重的两种打包格式，**架构敏感**——Ada/L4 上 PTQ1_0 快，H100/Blackwell 上 PQ2_0 快。

## 两种格式

### PTQ1_0：dense trit packing

5 个 trit 打包进 8 bit。

$$3^5 = 243 < 256 = 2^8$$

所以 5 个 trit（每个 $\in \{0, 1, 2\}$）可以编码成一个 base-3 整数（0..242），存进 1 byte。

$$\text{packed} = t_0 \cdot 81 + t_1 \cdot 27 + t_2 \cdot 9 + t_3 \cdot 3 + t_4$$

- **密度**：$8/5 = 1.6$ bpw + scale 0.125 = **1.725 bpw**
- **解包**：需要 5 次除法 + 取模（base-3 解码）
- **优势**：最省空间

### PQ2_0：2-bit slot

每个 trit 占 2-bit slot。$\{−1, 0, +1\} \to \{0, 1, 2\}$，存进 2 bit（slot 3 unused）。
4 个 trit 打包进 1 byte。

$$\text{packed} = t_0 \,|\, (t_1 \ll 2) \,|\, (t_2 \ll 4) \,|\, (t_3 \ll 6)$$

- **密度**：$2 + 0.125 = 2.125$ bpw
- **解包**：只需 shift + mask（位运算）
- **优势**：解码简单，对齐友好

## 密度对比

| 格式 | bpw | 解包方式 |
|---|---|---|
| 1-bit (binary) | 1.125 | XNOR（最快） |
| **PTQ1_0** | **1.725** | 除法 + 取模（最紧） |
| **PQ2_0** | **2.125** | shift + mask（最简） |

PTQ1_0 比 PQ2_0 省 0.4 bpw——在 27B 模型上差 ~2 GB。

## 架构敏感性

为什么不同硬件上最优格式不同？

### Ada / L4：PTQ1_0 更快

这些架构的 trit kernel 优化得好。除法可以用查表 + 乘法近似，
密集的 packing 让 memory bandwidth 利用率高。

### H100 / Blackwell：PQ2_0 更快

这些架构的 2-bit slot 对齐更好。H100 的 Tensor Core 对 2-bit 边界友好，
shift + mask 的解码延迟比除法低，弥补了体积大的劣势。

### Apple Silicon：1-bit XNOR 优势小

ARM NEON 的 SIMD 宽度（128 bit）不如 x86 AVX2（256 bit），
位运算的硬件优势被抵消。

!!! insight "见解：没有 universally 最优的 packing"
    这说明"最优"是 **workload × 硬件** 的函数，不是绝对的。
    Bonsai 提供两种格式，让用户按硬件选——这本身就是一个设计决策：
    不押注单一架构，而是适配生态。

## 代码

```python
from bonsai import pack_trits_ptq10, unpack_trits_ptq10, pack_trits_pq20, unpack_trits_pq20

trits = torch.randint(-1, 2, (40,))  # {−1, 0, +1}

# PTQ1_0: 5 trit → 8 bit
packed = pack_trits_ptq10(trits)      # uint8, 8 bytes
unpacked = unpack_trits_ptq10(packed)  # 还原
assert (trits == unpacked).all()

# PQ2_0: 4 trit → 8 bit
packed = pack_trits_pq20(trits)       # uint8, 10 bytes
unpacked = unpack_trits_pq20(packed)
assert (trits == unpacked).all()
```

## group scale

两种格式都配 group scale：1 FP16 (16 bit) per 128 weights = 0.125 bpw。

scale 是 FP16，在 matmul 时乘到结果上：

$$y = \text{ternary\_matmul}(x, \text{trit}) \cdot \text{scale}$$

## 实验验证

`experiments/06_xnor_popcount_packing.py` Part 3 + Part 5：

- 密度计算：1.125 / 1.725 / 2.125 bpw
- packing roundtrip：误差 = 0（两种格式都精确还原）

---

详见 `bonsai/kernel.py`。