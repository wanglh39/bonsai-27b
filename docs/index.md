# Bonsai-27B 教学复现

> 用 PyTorch 从零复现 prism-ml/Bonsai-27B 的**核心算子**与**设计哲学**，
> 重点讲清"为什么它敢做到 1.7 bit 还不崩"。

## 这是什么

Bonsai-27B 把 27B 参数的大模型压到 **3.9 GB**（1-bit）或 **5.9 GB**（ternary），
还能保留 **90–98%** 的 FP16 能力。传统量化在 sub-4-bit 会"塌陷"（math 掉到 57 分），Bonsai 不会。

支点是一个反直觉的机制：

> **Hadamard 旋转**——对权重做正交变换把"难量化的各向异性权重"变成"好量化的各向同性权重"，代价是零额外 bit。

本包用 PyTorch 复现五个核心算子 + 完整 Transformer 架构，配 6 个对比实验让你亲眼看到每个支点的作用。

## 五个核心算子

| 算子 | 文件 | 一句话原理 |
|---|---|---|
| `BlockHadamard` | `bonsai/hadamard.py` | \(W' = H \cdot W \cdot H^\top\)，正交变换打散 outlier，零额外 bit |
| `ternarize` / `binarize` | `bonsai/quantize.py` | 权重 → {−1,0,+1} 或 {−1,+1}，每 128 个共享一个 FP16 scale |
| `HadamardTernaryLinear` | `bonsai/ops.py` | 完整低 bit Linear 层：旋转 activation → 低 bit 乘 → 反旋转输出 |
| `xnor_popcount_matmul` | `bonsai/kernel.py` | 1-bit matmul 等价于 XNOR+popcount 位运算，不需乘法器 |
| `BlockDenoisingDrafter` | `bonsai/dspark.py` | 一次前向草 k 个 token（vs 自回归 k 次），diffusion × spec decoding |

## 设计哲学一句话

**Intelligence Density**：\(D = -\log_2(1 - \text{score}/100) / \text{size\_GB}\)

端侧 AI 的真目标是"**每 GB 的可用智力**"，不是绝对精度。
FP16 的 85 分在 54 GB 里是浪费——装不下 = 0 分。
Bonsai 的全部设计都在最大化这个密度。

## 快速开始

```bash
pip install -r requirements.txt

# 跑 6 个实验
python experiments/01_hadamard_magic.py        # 旋转增大权重误差但减小下游误差
python experiments/02_ternary_vs_binary.py     # 端到端 MiniBonsai 前向对比
python experiments/03_intelligence_density.py  # 智力密度度量
python experiments/04_qat_the_real_gain.py     # QAT 才是 98% 能力的来源
python experiments/05_dspark_block_denoising.py # DSpark block-denoising drafter
python experiments/06_xnor_popcount_packing.py  # XNOR-popcount + packing
```

## 文档导读

- **[设计哲学](philosophy/intelligence-density.md)**：Intelligence Density、为什么不崩、算法创新谱系
- **[核心算子](operators/hadamard.md)**：每个算子的底层原理、数学推导、实现细节
- **[模型架构](architecture/transformer.md)**：RoPE + KV cache + hybrid attention
- **[实验解读](experiments/guide.md)**：6 个实验各自展现什么、怎么读结果
- **[API 参考](api/reference.md)**：完整 API 文档

## 技术栈来源

Bonsai-27B 不是凭空出现的，它站在三个巨人的肩膀上：

| 技术 | 来源 | Bonsai 的贡献 |
|---|---|---|
| Hadamard-domain quantization | QuIP# (Chee et al.) | 端到端折叠进存储，零额外 bit |
| Ternary weight networks | BitNet 1.58b (Ma et al.) | 配合旋转 + group scale + QAT |
| DSpark speculative decoding | DeepSeek DSpark | 适配低 bit target，block-denoising drafter |
| Hybrid attention | Qwen3.6 | 75% linear / 25% full，长上下文省显存 |

## 进一步阅读

- 白皮书：`github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf`
- 模型卡片：`huggingface.co/prism-ml/Bonsai-27B-gguf`
- 理论源头：QuIP#（Hadamard-domain quantization）、BitNet 1.58b（ternary weight networks）