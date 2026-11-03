"""Bonsai-27B 教学复现：低 bit 量化的核心算子 + 完整模型架构。

本包复现 prism-ml/Bonsai-27B 的核心机制：
  1. Hadamard 旋转基  (bonsai.hadamard)
  2. 三值化/二值化     (bonsai.quantize)
  3. 低 bit matmul    (bonsai.ops)
  4. XNOR/popcount kernel + packing (bonsai.kernel)
  5. DSpark drafter   (bonsai.dspark)
  6. Transformer 架构  (bonsai.block) —— RoPE + KV cache + hybrid attention

设计哲学一句话：
    "用正交变换（Hadamard）把难量化的权重变成好量化的权重，
     代价是零额外 bit——这是 Bonsai 敢做到 1.7 bit 还不崩的支点。"
"""
from .hadamard import hadamard_matrix, random_hadamard, BlockHadamard
from .quantize import ternarize, binarize, dequantize
from .ops import ternary_linear, binary_linear, HadamardTernaryLinear
from .kernel import (
    xnor_popcount_matmul,
    ternary_matmul_via_xnor,
    packing_bpw,
    pack_trits_ptq10,
    unpack_trits_ptq10,
    pack_trits_pq20,
    unpack_trits_pq20,
)
from .dspark import (
    TargetModel,
    AutoregressiveDrafter,
    BlockDenoisingDrafter,
    verify,
    train_drafter,
    make_sequence,
)
from .block import (
    RMSNorm,
    RoPE,
    KVCache,
    SwiGLUMLP,
    SimpleAttention,
    LinearAttention,
    SimpleBlock,
    MiniBonsai,
)

__all__ = [
    # hadamard
    "hadamard_matrix",
    "random_hadamard",
    "BlockHadamard",
    # quantize
    "ternarize",
    "binarize",
    "dequantize",
    # ops
    "ternary_linear",
    "binary_linear",
    "HadamardTernaryLinear",
    # kernel
    "xnor_popcount_matmul",
    "ternary_matmul_via_xnor",
    "packing_bpw",
    "pack_trits_ptq10",
    "unpack_trits_ptq10",
    "pack_trits_pq20",
    "unpack_trits_pq20",
    # dspark
    "TargetModel",
    "AutoregressiveDrafter",
    "BlockDenoisingDrafter",
    "verify",
    "train_drafter",
    "make_sequence",
    # block
    "RMSNorm",
    "RoPE",
    "KVCache",
    "SwiGLUMLP",
    "SimpleAttention",
    "LinearAttention",
    "SimpleBlock",
    "MiniBonsai",
]