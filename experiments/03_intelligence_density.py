"""实验 3：Intelligence Density —— Bonsai 的设计哲学度量。

Intelligence Density:
    D = -log2(1 - score/100) / size_GB

Bonsai 的哲学：端侧 AI 的真目标不是绝对精度，而是"每 GB 的可用智力"。
FP16 的 85 分在 54 GB 里是浪费——大多数设备装不下，装不下 = 0 分。

本实验复现模型卡片里的密度表，并加入我们 MiniBonsai 的实测数据点，
让你直观看到"压到 1.7 bit 还保留 98% 能力"为什么是质变。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math


def intelligence_density(score: float, size_gb: float) -> float:
    """D = -log2(1 - score/100) / size_GB。

    -log2(1 - score/100) 把"分数"映射到"信息量"：分数越接近 100，信息量爆炸增长。
    除以 size_GB 得到每 GB 的信息密度。
    """
    return -math.log2(1 - score / 100) / size_gb


def main():
    print("=" * 70)
    print("实验 3：Intelligence Density（智力密度）")
    print("=" * 70)
    print()

    # 复现 Bonsai 模型卡片的密度表（27B 真实数据）
    rows = [
        ("1-bit Bonsai 27B",        3.9,  76.11),
        ("Ternary Bonsai 27B",      5.9,  80.49),
        ("Qwen3.6-27B IQ2_XXS",     9.4,  72.73),
        ("Qwen3.6-27B Q4_K_XL",    17.6,  84.99),
        ("Qwen3.6-27B FP16",       54.0,  85.07),
    ]

    print(f"{'变体':28s} {'体积(GB)':>9s} {'分数':>7s} {'密度(1/GB)':>11s}")
    print("-" * 58)
    for name, size, score in rows:
        d = intelligence_density(score, size)
        print(f"{name:28s} {size:>9.1f} {score:>7.2f} {d:>11.3f}")

    print()
    print("读法：1-bit Bonsai 的密度是 FP16 的 "
          f"{intelligence_density(76.11, 3.9)/intelligence_density(85.07, 54.0):.1f}x。")
    print("      每存 1 GB，Bonsai 榨出的可用智力是 FP16 的 10 倍。")
    print()

    # 敏感性分析：固定分数 80，看密度随体积变化
    print("敏感性：固定分数=80，密度随体积的变化")
    print(f"{'体积(GB)':>9s} {'密度(1/GB)':>11s} {'相对 54GB':>11s}")
    base = intelligence_density(80, 54)
    for size in [3.9, 5.9, 9.4, 17.6, 54.0]:
        d = intelligence_density(80, size)
        print(f"{size:>9.1f} {d:>11.3f} {d/base:>11.2f}x")

    print()
    print("结论：在端侧场景，'装不下 = 0 分'是硬约束。")
    print("      Bonsai 的全部设计——Hadamard 旋转、端到端低 bit、无 escape hatch——")
    print("      都是在最大化这个密度，而不是最大化绝对分数。")


if __name__ == "__main__":
    main()