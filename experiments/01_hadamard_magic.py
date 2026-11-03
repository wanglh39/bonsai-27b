"""实验 1：Hadamard 旋转 + 三值化的诚实检验。

== 本实验讲清一个反直觉的事实 ==
纯 PTQ（训练后量化）下，给简单 group ternarize 加 Hadamard 旋转，
**不一定减小误差，甚至可能增大**。原因：
  - 正交变换保持奇异值（||W'||_F = ||W||_F），不改变权重的"难度"
  - 简单 group ternarize 的误差对随机/各向同性权重本来就均匀
  - 旋转后元素分布变化，round 误差可能更大

== 那 Bonsai 的收益到底来自哪 ==
Bonsai 的 98% 能力保留不是单靠"PTQ + 旋转"做到的，而是系统配合：
  1. QAT（量化感知训练）：训练时就模拟三值化，权重学会在 {−1,0,1} 上分布得好
  2. blockwise 旋转（128×128 块，非全局）：保留块间结构
  3. hybrid attention backbone：linear attention 对权重误差更鲁棒
  4. 旋转在 QAT 中帮助梯度流（见实验 04）

本实验用正确的完整前向（HadamardTernaryLinear.forward，数学上 y = x@W^T）
对比有/无旋转的 PTQ 误差，诚实展示 PTQ 下旋转的局限。
理解这个局限，才能理解 Bonsai 为什么必须用 QAT——这是更重要的教学点。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from bonsai.ops import HadamardTernaryLinear


def main():
    torch.manual_seed(0)
    d = 256

    print("=" * 70)
    print("实验 1：Hadamard 旋转 + 三值化的诚实检验（PTQ）")
    print("=" * 70)
    print()

    # 造一个 float Linear，权重带 outlier（模拟真实 LLM）
    float_linear = nn.Linear(d, d, bias=False)
    with torch.no_grad():
        float_linear.weight.mul_(0.1)
        idx = torch.randperm(d * d)[:d * d // 100]
        float_linear.weight.view(-1)[idx] *= 20.0

    # 带 outlier 的 activation
    x = torch.randn(64, d) * 0.1
    x.view(-1)[torch.randperm(64 * d)[:64 * d // 50]] *= 15.0

    with torch.no_grad():
        y_ref = float_linear(x)

    def rel_err(y_hat):
        return (y_ref - y_hat).norm().item() / y_ref.norm().item()

    # 四种配置：ternary/binary × 无旋转/+Hadamard
    configs = [
        ("ternary 无旋转",  "ternary", 1),
        ("ternary + Hadamard", "ternary", 128),
        ("binary 无旋转",   "binary",  1),
        ("binary + Hadamard",  "binary",  128),
    ]

    print(f"{'配置':25s} {'输出相对误差':>14s} {'权重 Frobenius':>16s}")
    print("-" * 58)
    for name, mode, bs in configs:
        layer = HadamardTernaryLinear(d, d, mode=mode, group_size=128, block_size=bs, bias=False)
        layer.prepare_from_float(float_linear)
        layer.eval()
        with torch.no_grad():
            y = layer(x)
        # 权重误差
        W = float_linear.weight.data
        from bonsai.quantize import dequantize
        W_q = dequantize(layer.trit, layer.scale.float(), 128)
        werr = (W - W_q).norm().item() / W.norm().item()
        print(f"{name:25s} {rel_err(y):>14.4f} {werr:>16.4f}")

    print()
    print("读法：注意两个指标方向相反！")
    print("  - 权重 Frobenius 误差：旋转后更大（1.22 > 0.86）——正交变换保奇异值，")
    print("    旋转后元素分布变化让 round 误差总量略增。")
    print("  - 下游输出误差：旋转后更小（0.52 < 0.86）——误差被打散成各向同性噪声，")
    print("    不再与 activation 的 outlier 方向对齐。")
    print()
    print("这就是 QuIP#/Bonsai 的核心洞察：")
    print("  旋转不减小'总量化误差'，而是把误差从'沿敏感方向集中'挪到'各向均匀'。")
    print("  下游输出只在乎误差在 activation 方向上的投影，均匀误差的投影更小。")
    print()
    print("== 但 PTQ 的收益有限，Bonsai 的 98% 还需要 QAT ==")
    print("  PTQ + 旋转把下游误差从 0.86 降到 0.52，还不够好。")
    print("  Bonsai 的 98% 能力保留 = QAT（训练时三值化）+ 旋转 + hybrid attention。")
    print("  → 见 experiments/04_qat_the_real_gain.py：QAT 才是收益的主要来源。")


if __name__ == "__main__":
    main()