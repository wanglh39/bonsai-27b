"""三值化 / 二值化算子 —— Bonsai 的权重表示层。

== 三值化 (Ternary, Bonsai-2 / Ternary-Bonsai) ==
    权重 ∈ {−1, 0, +1}，每 group_size 个权重共享一个 FP16 scale α。
    一个 trit 承载 log₂3 ≈ 1.585 bit 信息，
    加 16-bit scale 摊到 128 个权重上 → 有效 ~1.71 bits/weight。

    赋值规则（BitNet 1.58b 路线）：
        α_g = mean(|w_g|)                    # group scale，组的平均幅度
        Δ_g = 0.7 * α_g                       # 阈值，低于它判 0
        w̃ = round(w / Δ_g) clip 到 {−1,0,1}   # 量化
        反量化：ŵ = w̃ * Δ_g  （或 ŵ = w̃ * α_g，看用哪个做 scale）

    为什么阈值是 0.7*mean(|w|)：经验上让 ~一半权重为 0（稀疏），
    非零权重保留符号和量级主信息。0 带来稀疏 → matmul 可跳过。

== 二值化 (Binary, 1-bit Bonsai) ==
    权重 ∈ {−1, +1}，无 0。更激进：幅度信息全交给 group scale。
        α_g = mean(|w_g|)
        w̃ = sign(w)
        反量化：ŵ = w̃ * α_g
    有效 1.125 bits/weight（1 sign bit + 16-bit scale / 128）。

== group_size=128 为什么 ==
    group 越小 → scale 越密 → 精度越高但 scale 开销越大。
    128 让 scale 开销 16/128 = 0.125 bit 压到可忽略，是甜点。
    Bonsai 模型卡片：Q1_0_g128 / ternary g128 的 "g128" 即此。
"""
from __future__ import annotations

import torch
from torch import Tensor


def ternarize(
    W: Tensor,
    group_size: int = 128,
    threshold_ratio: float = 0.7,
    return_scale: bool = True,
) -> tuple[Tensor, Tensor] | Tensor:
    """三值化权重 W ∈ R^{...} 到 {−1, 0, +1} + group scale。

    参数：
        W: 任意形状权重
        group_size: 共享一个 scale 的权重数，默认 128（Bonsai 的 g128）
        threshold_ratio: 判 0 的阈值比例，默认 0.7（BitNet 1.58b 经验值）
        return_scale: True 返回 (trit, scale)，False 只返回 trit

    返回：
        trit: int8 张量，值 ∈ {−1, 0, +1}，形状同 W
        scale: float 张量，每个 group 一个 scale，形状为 (num_groups,)

    教学要点：trit 用 int8 存储只是为了 PyTorch 方便；
    真实打包里 5 个 trit 塞进 8 bit（PTQ1_0 dense packing），
    或每个 trit 占 2-bit slot（PQ2_0），由 kernel 直接消费。
    """
    orig_shape = W.shape
    # 把权重摊平后按 group_size 切组
    W_flat = W.reshape(-1)
    n = W_flat.numel()
    pad = (-n) % group_size
    if pad:
        W_flat = torch.nn.functional.pad(W_flat, (0, pad))
    W_grouped = W_flat.view(-1, group_size)  # (num_groups, group_size)

    # group scale α = mean(|w|)，对应 Bonsai 的 FP16 group-wise scaling
    scale = W_grouped.abs().mean(dim=-1, keepdim=True)  # (num_groups, 1)
    eps = 1e-8
    # 阈值 Δ = threshold_ratio * α，归一化后 |w/Δ| < 1 → 判 0
    threshold = threshold_ratio * scale
    W_norm = W_grouped / (threshold + eps)

    # round 到最近整数，再 clip 到 {−1, 0, +1}
    trit = W_norm.round().clamp(-1.0, 1.0).to(torch.int8)  # (num_groups, group_size)

    # 去掉 padding
    trit = trit.reshape(-1)[:n].reshape(orig_shape)
    scale = scale.squeeze(-1)  # (num_groups,)

    if return_scale:
        return trit, scale
    return trit


def binarize(
    W: Tensor,
    group_size: int = 128,
    return_scale: bool = True,
) -> tuple[Tensor, Tensor] | Tensor:
    """二值化权重 W 到 {−1, +1} + group scale（1-bit Bonsai 路线）。

    比 ternarize 更激进：去掉 0，幅度信息全交给 scale。
    反量化 ŵ = sign(w) * α_g，有效 1.125 bits/weight。
    """
    orig_shape = W.shape
    W_flat = W.reshape(-1)
    n = W_flat.numel()
    pad = (-n) % group_size
    if pad:
        W_flat = torch.nn.functional.pad(W_flat, (0, pad))
    W_grouped = W_flat.view(-1, group_size)

    scale = W_grouped.abs().mean(dim=-1, keepdim=True)
    sign = torch.sign(W_grouped)
    # 0 权重随机给个符号，避免全 0 group
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    sign = sign.to(torch.int8)

    sign = sign.reshape(-1)[:n].reshape(orig_shape)
    scale = scale.squeeze(-1)

    if return_scale:
        return sign, scale
    return sign


def dequantize(trit_or_sign: Tensor, scale: Tensor, group_size: int = 128) -> Tensor:
    """反量化：把 (trit/sign, scale) 还原成近似的 float 权重。

    ŵ = w̃ * α_g（每个 group 内乘同一个 scale）。
    教学用：真实推理从不解压，kernel 直接在打包格式上做 matmul（见 ops.py）。
    """
    orig_shape = trit_or_sign.shape
    flat = trit_or_sign.reshape(-1).to(torch.float32)
    n = flat.numel()
    pad = (-n) % group_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    flat = flat.view(-1, group_size)
    # scale 形状 (num_groups,) → (num_groups, 1)
    flat = flat * scale.unsqueeze(-1)
    flat = flat.reshape(-1)[:n].reshape(orig_shape)
    return flat