"""1-bit XNOR/popcount kernel + PTQ1_0/PQ2_0 packing —— Bonsai 的硬件层。

== 核心洞察 ==
1-bit 权重 {−1,+1} 的 matmul 不需要乘法器，只需 XNOR + popcount（位运算）。
  理论上：CPU SIMD XNOR 一次处理 64/128/256 bit，vs FP16 FMA 一次 1 个。
  这是 1-bit 模型在端侧快的根本原因。

三值化 {−1,0,+1} 有 0，不能纯 XNOR。Bonsai 两种 packing：
  - PTQ1_0：dense trit packing，5 trit → 8 bit (3^5=243<256=2^8)，1.725 bpw
  - PQ2_0：2-bit slot，显式编码 3 值（+1 unused slot），2.125 bpw
  架构敏感：Ada/L4 上 PTQ1_0 快，H100/Blackwell 上 PQ2_0 快
"""
from __future__ import annotations

import torch
from torch import Tensor


def xnor_popcount_matmul(a: Tensor, b: Tensor) -> Tensor:
    """1-bit XNOR-popcount matmul。a: (m,k), b: (k,n), 都 ∈ {−1,+1}。

    原理：把 {−1,+1} 映射到 {0,1}，乘法等价于 XNOR（相同=1，不同=0）。
      a[i,k]*b[k,j] = +1 if a==b else -1
      dot = 2*popcount(XNOR(a,b)) - k

    硬件：XNOR 和 popcount 都是单周期位运算，不需要乘法器。
    本函数用整数比较模拟（真实硬件用位运算 pack/unpack，但数学等价）。
    """
    a_bits = (a > 0).to(torch.int32)
    b_bits = (b > 0).to(torch.int32)
    xnor = (a_bits.unsqueeze(2) == b_bits.unsqueeze(0)).to(torch.int32)
    popcount = xnor.sum(dim=1)
    k = a.shape[1]
    return (2 * popcount - k).to(torch.float32)


def ternary_matmul_via_xnor(a: Tensor, b: Tensor) -> Tensor:
    """三值化 matmul via XNOR + mask。a: (m,k) ∈ {−1,+1}, b: (k,n) ∈ {−1,0,+1}。

    b 有 0 → 不能纯 XNOR。拆成 nonzero_mask + sign，只在非零位置做 XNOR-popcount。
    额外开销 vs 纯 1-bit：
      1. mask 计算 + gather（非零位置不连续 → 不规则内存访问）
      2. 每列非零数不同 → load imbalance
      3. 需要存 magnitude（0/1）→ 额外 bit
    """
    m = a.shape[0]
    n = b.shape[1]
    out = torch.zeros(m, n)
    for j in range(n):
        col = b[:, j]
        nz_mask = col != 0
        k_nz = nz_mask.sum().item()
        if k_nz == 0:
            continue
        a_nz = a[:, nz_mask]
        b_sign = torch.sign(col[nz_mask].float())
        a_bits = (a_nz > 0).to(torch.int32)
        b_bits = (b_sign > 0).to(torch.int32)
        xnor = (a_bits == b_bits.unsqueeze(0)).to(torch.int32)
        popcount = xnor.sum(dim=1)
        out[:, j] = 2 * popcount - k_nz
    return out


def packing_bpw() -> dict[str, float]:
    """计算不同 packing 的实际 bpw (bits per weight)。

    group scale: 1 FP16 (16 bit) per 128 weights = 0.125 bpw

    1-bit (binary): 1 + 0.125 = 1.125 bpw
    PTQ1_0 (dense trit): 5 trit in 8 bit = 1.6 + 0.125 = 1.725 bpw
    PQ2_0 (2-bit slot): 2 + 0.125 = 2.125 bpw
    """
    scale_bpw = 16 / 128
    return {
        "1-bit (binary)": 1 + scale_bpw,
        "PTQ1_0 (dense trit)": 8 / 5 + scale_bpw,
        "PQ2_0 (2-bit slot)": 2 + scale_bpw,
    }


def pack_trits_ptq10(trits: Tensor) -> Tensor:
    """PTQ1_0 packing：5 个 trit 打包进 8 bit。

    3^5 = 243 < 256 = 2^8，所以 5 个 trit（每个 ∈ {−1,0,+1}）可以
    编码成一个 base-3 整数（0..242），存进 1 byte。

    输入：trits (..., ) ∈ {−1,0,+1}，长度必须是 5 的倍数
    输出：uint8 (..., /5)
    """
    flat = trits.reshape(-1)
    assert flat.numel() % 5 == 0, "trit 数量必须是 5 的倍数"
    t = (flat + 1).to(torch.int32)  # {0,1,2}
    t = t.view(-1, 5)
    packed = t[:, 0] * 81 + t[:, 1] * 27 + t[:, 2] * 9 + t[:, 3] * 3 + t[:, 4]
    return packed.to(torch.uint8)


def unpack_trits_ptq10(packed: Tensor) -> Tensor:
    """PTQ1_0 unpacking：8 bit 解包出 5 个 trit。"""
    p = packed.to(torch.int32).reshape(-1)
    t4 = p % 3; p = p // 3
    t3 = p % 3; p = p // 3
    t2 = p % 3; p = p // 3
    t1 = p % 3; p = p // 3
    t0 = p
    out = torch.stack([t0, t1, t2, t3, t4], dim=1).reshape(-1)
    return (out - 1).to(torch.int8)


def pack_trits_pq20(trits: Tensor) -> Tensor:
    """PQ2_0 packing：每个 trit 占 2-bit slot。

    {−1,0,+1} → {0,1,2}，存进 2 bit（slot 3 unused）。
    4 个 trit 打包进 1 byte。
    """
    flat = trits.reshape(-1)
    assert flat.numel() % 4 == 0, "trit 数量必须是 4 的倍数"
    t = (flat + 1).to(torch.int32)  # {0,1,2}
    t = t.view(-1, 4)
    packed = t[:, 0] | (t[:, 1] << 2) | (t[:, 2] << 4) | (t[:, 3] << 6)
    return packed.to(torch.uint8)


def unpack_trits_pq20(packed: Tensor) -> Tensor:
    """PQ2_0 unpacking：2-bit slot 解包。"""
    p = packed.to(torch.int32).reshape(-1)
    t0 = p & 0x3
    t1 = (p >> 2) & 0x3
    t2 = (p >> 4) & 0x3
    t3 = (p >> 6) & 0x3
    out = torch.stack([t0, t1, t2, t3], dim=1).reshape(-1)
    return (out - 1).to(torch.int8)