# Bonsai-27B 教学复现

用 PyTorch 复现 prism-ml/Bonsai-27B 的**核心算子**，重点讲清"为什么它敢做到 1.7 bit 还不崩"。

📖 **[完整文档站](https://wanglh39.github.io/bonsai-27b/)** —— 哲学、算子推导、架构、实验解读、API 参考

## 它在讲什么

Bonsai-27B 把 27B 大模型压到 3.9 GB（1-bit）或 5.9 GB（ternary），还能保留 90–98% 的 FP16 能力。
传统量化在 sub-4-bit 会"塌陷"（math 掉到 57 分），Bonsai 不会。支点是一个反直觉的机制：

> **Hadamard 旋转**——对权重做正交变换把"难量化的各向异性权重"变成"好量化的各向同性权重"，代价是零额外 bit。

本包复现五个核心算子 + 完整 Transformer 架构，配对比实验让你亲眼看到每个支点的作用。

## 项目结构

```
bonsai/
  hadamard.py    # 核心①：Hadamard 旋转基（Bonsai 不塌陷的真正原因）
  quantize.py    # 核心②：三值化 {−1,0,+1} / 二值化 {−1,+1} + group scale
  ops.py         # 核心③：低 bit matmul + HadamardTernaryLinear 完整层
  kernel.py      # 核心④：XNOR-popcount kernel + PTQ1_0/PQ2_0 packing encode/decode
  dspark.py      # 核心⑤：DSpark block-denoising drafter（diffusion × spec decoding）
  block.py       # Transformer 架构：RoPE + KV cache + hybrid attention (75% linear / 25% full)
experiments/
  01_hadamard_magic.py       # 核心：旋转增大权重误差但减小下游误差（QuIP# 洞察）
  02_ternary_vs_binary.py    # 端到端 MiniBonsai 前向对比
  03_intelligence_density.py # Bonsai 的设计哲学度量
  04_qat_the_real_gain.py    # QAT 才是 Bonsai 98% 能力的真正来源
  05_dspark_block_denoising.py  # DSpark：diffusion × speculative decoding 的交叉创新
  06_xnor_popcount_packing.py   # 1-bit XNOR/popcount kernel + PTQ1_0/PQ2_0 packing
```

## 跑

```bash
pip install -r requirements.txt
python experiments/01_hadamard_magic.py
python experiments/02_ternary_vs_binary.py
python experiments/03_intelligence_density.py
python experiments/04_qat_the_real_gain.py
python experiments/05_dspark_block_denoising.py
python experiments/06_xnor_popcount_packing.py
```

## 五个算子的原理一句话版

| 算子 | 原理 |
|---|---|
| `ternarize` | 权重 → {−1,0,+1}，每 128 个共享一个 FP16 scale，有效 1.71 bpw |
| `binarize` | 权重 → {−1,+1}，幅度全交给 scale，有效 1.125 bpw |
| `BlockHadamard` | W' = H·W·Hᵀ，正交变换打散 outlier，零额外 bit |
| `xnor_popcount_matmul` | 1-bit matmul 等价于 XNOR+popcount 位运算，不需乘法器 |
| `BlockDenoisingDrafter` | 一次前向草 k 个 token（vs 自回归 k 次），diffusion × spec decoding |

## 实验揭示的三个层次（重要）

**实验 1**：Hadamard 旋转**增大权重 Frobenius 误差**（1.22 > 0.86）但**减小下游输出误差**（0.52 < 0.86）。
旋转不减小总误差，而是把误差从"沿敏感方向集中"挪到"各向均匀"——下游只在乎误差在 activation 方向的投影。

**实验 2**：纯 PTQ 下旋转收益不稳定（依赖权重的 outlier 结构）。随机初始化的权重没 outlier，旋转没好处。

**实验 4**：**QAT 才是 Bonsai 98% 能力的来源**。PTQ 损失 +0.90，QAT 损失只 +0.32。
Bonsai 不是"训练完再压"，而是"在压缩约束下训练"（STE + 三值化前向）。旋转在 QAT 中辅助梯度流，但 QAT 是基础。

**实验 5**：DSpark block-denoising drafter——Bonsai 独创的 diffusion × speculative decoding 交叉。
block drafter 一次前向草 k 个 token（vs 自回归的 k 次），code-like 接受率 0.96、chat-like 0.29，
展现 DSpark 实测的"code/math 2x、chat 收益少"（workload-dependent）。

**实验 6**：1-bit XNOR-popcount matmul 是浮点 matmul 的精确等价（误差 0），
但三值化 {−1,0,+1} 的 0 需要额外 mask（不能纯 XNOR）。PTQ1_0 (1.725 bpw) 比 PQ2_0 (2.125 bpw) 省空间但解码复杂，
架构敏感：Ada/L4 上 PTQ1_0 快，H100/Blackwell 上 PQ2_0 快。

## 设计哲学

**Intelligence Density**：`D = -log₂(1 - score/100) / size_GB`

端侧 AI 的真目标是"每 GB 的可用智力"，不是绝对精度。FP16 的 85 分在 54 GB 里是浪费——装不下 = 0 分。
Bonsai 的全部设计都在最大化这个密度。详见 `experiments/03`。

## 进一步阅读

- **[完整文档站](https://wanglh39.github.io/bonsai-27b/)** —— 哲学、算子推导、架构、实验解读、API 参考
- 白皮书：`github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf`
- 模型卡片：`huggingface.co/prism-ml/Bonsai-27B-gguf`
- 理论源头：QuIP#（Hadamard-domain quantization）、BitNet 1.58b（ternary weight networks）