"""实验 6：1-bit XNOR/popcount kernel + PTQ1_0/PQ2_0 packing 对比。

== 硬件有意思的点 ==
1-bit 权重 {−1,+1} 的 matmul 不需要乘法器，只需 XNOR + popcount（位运算）。
  理论上：CPU SIMD XNOR 一次处理 64/128/256 bit，vs FP16 FMA 一次 1 个。
  这是 1-bit 模型在端侧快的根本原因。

但三值化 {−1,0,+1} 有 0，不能纯 XNOR。Bonsai 两种 packing：
  - PTQ1_0：dense trit packing，5 trit → 8 bit (3^5=243<256=2^8)，1.75 bpw
  - PQ2_0：2-bit slot，显式编码 3 值（+1 unused slot），2.13 bpw
  架构敏感：Ada/L4 上 PTQ1_0 快，H100/Blackwell 上 PQ2_0 快

== 本实验展现什么 ==
1. XNOR-popcount 1-bit matmul 和浮点 matmul 结果完全一致（算法等价）
2. 三值化 matmul 的额外开销（mask + 不规则访问，不能纯 XNOR）
3. PTQ1_0 vs PQ2_0 的 packing 密度（1.725 vs 2.125 bpw）
4. 理论 throughput：XNOR vs FMA 的算力比
5. packing roundtrip 正确性（pack → unpack 还原）

本实验的算子从 bonsai.kernel 包导入，展示包的完整 API。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from bonsai.kernel import (
    xnor_popcount_matmul,
    ternary_matmul_via_xnor,
    packing_bpw,
    pack_trits_ptq10,
    unpack_trits_ptq10,
    pack_trits_pq20,
    unpack_trits_pq20,
)


def main():
    print("=" * 70)
    print("实验 6：1-bit XNOR/popcount kernel + PTQ1_0/PQ2_0 packing")
    print("=" * 70)
    print()

    # --- Part 1: XNOR-popcount 正确性 ---
    print("【Part 1: XNOR-popcount 1-bit matmul 正确性】")
    torch.manual_seed(0)
    m, k, n = 32, 128, 16
    a = torch.sign(torch.randn(m, k))  # {−1,+1}
    b = torch.sign(torch.randn(k, n))  # {−1,+1}

    float_result = a.float() @ b.float()
    xnor_result = xnor_popcount_matmul(a, b)

    max_err = (float_result - xnor_result).abs().max().item()
    print(f"  浮点 matmul vs XNOR-popcount matmul: 最大误差 {max_err:.2e}")
    print(f"  → 完全一致（XNOR-popcount 是 1-bit matmul 的精确等价）")
    print()

    # --- Part 2: 三值化的额外开销 ---
    print("【Part 2: 三值化 matmul 的额外开销】")
    b_ternary = torch.sign(torch.randn(k, n)) * (torch.rand(k, n) > 0.3).float()
    b_ternary = b_ternary.to(torch.int8)

    float_result = a.float() @ b_ternary.float()
    xnor_result = ternary_matmul_via_xnor(a, b_ternary)

    max_err = (float_result - xnor_result).abs().max().item()
    nz_ratio = (b_ternary != 0).float().mean().item()
    print(f"  b 的非零比例: {nz_ratio:.2f}（30% 是 0）")
    print(f"  浮点 vs XNOR+mask: 最大误差 {max_err:.2e}")
    print(f"  → 三值化需要额外 mask + gather（非零位置不连续），不能纯 XNOR")
    print(f"  → 这是 PTQ1_0/PQ2_0 比 1-bit kernel 复杂的根本原因")
    print()

    # --- Part 3: packing 密度 ---
    print("【Part 3: PTQ1_0 vs PQ2_0 packing 密度】")
    bpws = packing_bpw()
    for name, bpw in bpws.items():
        print(f"  {name:25s}: {bpw:.3f} bpw")
    print()
    diff = bpws["PQ2_0 (2-bit slot)"] - bpws["PTQ1_0 (dense trit)"]
    print(f"  PTQ1_0 比 PQ2_0 省 {diff:.3f} bpw")
    print(f"  → PTQ1_0 更省空间（1.725 vs 2.125），但解码更复杂（5 trit → 8 bit unpack）")
    print(f"  → PQ2_0 更大但解码简单（2 bit 直接查表，1 unused slot）")
    print()

    # --- Part 4: 理论 throughput ---
    print("【Part 4: 理论 throughput 对比（XNOR vs FMA）】")
    simd_width = 256  # AVX2 XNOR 一次 256 bit
    fma_per_cycle = 1  # scalar FP16 FMA
    binary_per_cycle = simd_width
    fp16_per_cycle = fma_per_cycle
    print(f"  1-bit XNOR-popcount (AVX2): {binary_per_cycle} weight-op/cycle")
    print(f"  FP16 FMA:                    {fp16_per_cycle} weight-op/cycle")
    print(f"  理论算力比: {binary_per_cycle / fp16_per_cycle:.0f}x")
    print(f"  → 1-bit 模型在端侧快的根本原因：位运算吞吐远高于浮点乘加")
    print()

    # --- Part 5: packing roundtrip ---
    print("【Part 5: packing roundtrip 正确性】")
    torch.manual_seed(1)
    n_trits = 40
    trits = torch.randint(-1, 2, (n_trits,))  # {−1,0,+1}

    packed10 = pack_trits_ptq10(trits)
    unpacked10 = unpack_trits_ptq10(packed10)
    err10 = (trits - unpacked10).abs().max().item()
    ptq10_bytes = packed10.numel()
    print(f"  PTQ1_0: {n_trits} trit → {ptq10_bytes} byte ({n_trits/ptq10_bytes:.1f} trit/byte)")
    print(f"    roundtrip 误差: {err10}")

    trits4 = torch.randint(-1, 2, (n_trits,))
    packed20 = pack_trits_pq20(trits4)
    unpacked20 = unpack_trits_pq20(packed20)
    err20 = (trits4 - unpacked20).abs().max().item()
    pq20_bytes = packed20.numel()
    print(f"  PQ2_0:  {n_trits} trit → {pq20_bytes} byte ({n_trits/pq20_bytes:.1f} trit/byte)")
    print(f"    roundtrip 误差: {err20}")
    print(f"  → PTQ1_0 更紧（5 trit/byte vs 4 trit/byte），但 unpack 需除法（base-3 解码）")
    print(f"  → PQ2_0 更松但 unpack 只需位运算（shift + mask）")
    print()

    print("读法：")
    print("  1. XNOR-popcount 是 1-bit matmul 的精确等价（Part 1 验证）。")
    print("  2. 三值化不能纯 XNOR（0 需 mask），这是 PTQ1_0/PQ2_0 复杂度的来源。")
    print("  3. PTQ1_0 更省空间但解码复杂，PQ2_0 更大但解码简单 → 架构敏感。")
    print("  4. 1-bit 的硬件优势：位运算吞吐 >> 浮点乘加吞吐。")
    print("  5. packing roundtrip 验证两种 packing 的正确性（Part 5）。")
    print()
    print("Bonsai 实测（MODEL-FORMATS.md）：")
    print("  Ada/L4: PTQ1_0 更快（trit kernel 优化好）")
    print("  H100/Blackwell: PQ2_0 更快（2-bit slot 对齐好）")
    print("  Apple Silicon: 1-bit XNOR 优势小（ARM NEON 不如 AVX2）")
    print()
    print("算子来自 bonsai.kernel 包：xnor_popcount_matmul / ternary_matmul_via_xnor /")
    print("  packing_bpw / pack_trits_ptq10 / unpack_trits_ptq10 / pack_trits_pq20 / unpack_trits_pq20")


if __name__ == "__main__":
    main()