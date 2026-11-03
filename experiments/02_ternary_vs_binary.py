"""实验 2：三值化 vs 二值化 + 有/无 Hadamard，端到端跑 MiniBonsai。

对比四种配置在一个小 LM 前向任务上的输出差异：
  1. float baseline
  2. ternary 无旋转
  3. ternary + Hadamard
  4. binary + Hadamard

预期：
  - ternary+Hadamard 最接近 float
  - ternary 无旋转 明显偏离
  - binary+Hadamard 比 ternary 无旋转 还接近 float（旋转的收益 > 多一个 bit 的收益）

这验证 Bonsai 的设计取舍：旋转比 bit 数更重要。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from bonsai.block import MiniBonsai
from bonsai.ops import HadamardTernaryLinear


def output_divergence(y_ref: torch.Tensor, y: torch.Tensor) -> float:
    """输出 logits 的相对差异：||y - y_ref|| / ||y_ref||。"""
    return (y - y_ref).norm().item() / y_ref.norm().item()


def main():
    torch.manual_seed(42)
    vocab, dim, layers, heads, inter = 256, 128, 2, 4, 384
    # 维度取 128 的倍数，匹配 block_size=128

    print("=" * 70)
    print("实验 2：端到端 MiniBonsai 前向对比")
    print("=" * 70)

    # 1. float baseline
    model_f = MiniBonsai(vocab, dim, layers, heads, inter, low_bit=False)
    model_f.eval()

    # 随机输入
    tokens = torch.randint(0, vocab, (2, 16))
    with torch.no_grad():
        y_ref = model_f(tokens)
    print(f"float baseline 输出形状: {y_ref.shape}")
    print()

    def make_low_bit(mode: str, block_size: int) -> MiniBonsai:
        m = MiniBonsai(
            vocab, dim, layers, heads, inter,
            low_bit=True, mode=mode, group_size=128, block_size=block_size,
        )
        m.prepare_from_float(model_f)
        m.eval()
        return m

    configs = [
        ("ternary 无旋转",   "ternary", 1),   # block_size=1 → 退化为无旋转（H=1）
        ("ternary + Hadamard", "ternary", 128),
        ("binary 无旋转",    "binary",  1),
        ("binary + Hadamard",  "binary",  128),
    ]

    print(f"{'配置':25s} {'输出相对差异':>15s}")
    print("-" * 42)
    for name, mode, bs in configs:
        m = make_low_bit(mode, bs)
        with torch.no_grad():
            y = m(tokens)
        div = output_divergence(y_ref, y)
        print(f"{name:25s} {div:>15.4f}")

    print()
    print()
    print("读法：PTQ 下 +Hadamard 不一定更准（正交变换保奇异值，不改变 ternarize 总误差）。")
    print("      ternary 一致优于 binary（多一个 0 值带来稀疏 + 精度）。")
    print("      Bonsai 的 98% 能力保留来自 QAT + 系统配合，不是单纯 PTQ + 旋转。")
    print("      → 见 experiments/04_qat_the_real_gain.py：QAT 才是收益来源。")


if __name__ == "__main__":
    main()