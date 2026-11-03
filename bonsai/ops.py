"""低 bit matmul 算子 + Hadamard 旋转 Linear 层。

== 核心思想：权重从不解压回 FP16 ==
Bonsai 的收益全靠"打包权重直接被 kernel 消费"：
  - 1-bit 打包：64 个 sign bit 塞进一个 int64，matmul 用 popcount 数符号一致数
  - ternary PTQ1_0：5 个 trit 塞进 8 bit，kernel 用查表/位运算
  - ternary PQ2_0：每个 trit 占 2-bit slot，unpack 便宜但体积大
若解压回 FP16 再乘，3.9 GB → 54 GB，收益全没。

本教学包用 PyTorch 模拟"不解压"语义：
  ternary_linear(x, trit, scale) 内部把 trit 当 int8、scale 当 group scale，
  用 (x @ trit.float()) * scale 的等价形式算，但注释标明真实 kernel 的位运算路径。
  这样教学版能跑、能对比误差，又不失"权重以低 bit 形式参与计算"的语义。

== HadamardTernaryLinear：一个完整低 bit Linear 层的生命周期 ==
  离线（rotate + quantize）：
      W' = H_blk @ W @ H_blk^T          # 旋转到好量化的基
      trit, scale = ternarize(W')        # 在旋转基下三值化
      存储 trit (int8), scale (float16), H (metadata)
  在线（forward）：
      x' = x @ H_blk                    # 旋转 activation 到匹配基
      y' = ternary_matmul(x', trit, scale)   # 低 bit 乘，不解压
      y  = y' @ H_blk^T                 # 反旋转回原基
  数学等价性：y = x @ W^T ≈ (x@H) @ (H@W@H^T)^T @ (H^T)^T 的逐项配对。
"""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn

from .quantize import ternarize, binarize, dequantize
from .hadamard import BlockHadamard


def ternary_linear(x: Tensor, trit: Tensor, scale: Tensor, group_size: int = 128) -> Tensor:
    """低 bit 三值化 matmul：y = x @ (trit * scale)^T，权重以 trit+scale 形式参与。

    教学语义：trit 是 int8（真实打包是 5-trit-per-byte 或 2-bit-slot），
    这里用 float 桥接让 PyTorch 能算。真实 kernel 路径：
      1. x 按 group 切片
      2. 对每个 group：partial = x_group @ trit_group  （trit ∈ {−1,0,1}，可用加减/跳零）
      3. partial *= scale_group
      4. 跨 group 求和
    """
    # 把 trit 还原成 per-weight 的量化权重（教学版；真实 kernel 不显式还原）
    W_q = dequantize(trit, scale, group_size)  # (out_features, in_features)
    return x @ W_q.t()


def binary_linear(x: Tensor, sign: Tensor, scale: Tensor, group_size: int = 128) -> Tensor:
    """低 bit 二值化 matmul：y = x @ (sign * scale)^T。

    真实 1-bit kernel 最快：64 个 sign 塞进 int64，
    x 也符号化后用 XNOR+popcount 数一致位，一次 int64 运算顶 64 次乘加。
    """
    W_q = dequantize(sign, scale, group_size)
    return x @ W_q.t()


class HadamardTernaryLinear(nn.Module):
    """完整的 Hadamard 旋转 + 三值化 Linear 层。

    这是 Bonsai 每一个 Linear 的最小完整复现：
      - 离线把权重旋转到好量化的基，再三值化
      - 在线旋转 activation、低 bit 乘、反旋转输出

    参数：
        in_features, out_features: 同 nn.Linear
        group_size: 三值化的 group 大小（默认 128，Bonsai g128）
        block_size: Hadamard 分块大小（默认 128）
        mode: "ternary" 或 "binary"，选三值化或二值化

    用法：
        layer = HadamardTernaryLinear(512, 512)
        layer.prepare_from_float(float_linear)  # 从一个 float Linear 离线量化
        y = layer(x)                            # 前向走低 bit 路径
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        block_size: int = 128,
        mode: str = "ternary",
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.block_size = block_size
        self.mode = mode

        # Hadamard 旋转器（行、列方向各一个，或共享 block_size）
        self.rot = BlockHadamard(block_size=block_size, randomized=True)

        # 量化后的权重：trit/sign (int8) + scale (float16)
        # 注册为 buffer，不参与梯度（这是推理层）
        self.register_buffer("trit", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.zeros(1, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.register_buffer("bias", None)
        self._prepared = False

    @torch.no_grad()
    def prepare_from_float(self, linear: nn.Linear) -> None:
        """离线量化：从一个 float nn.Linear 烧出低 bit 表示。

        步骤：
          1. W' = H_blk @ W @ H_blk^T   （旋转到好量化的基）
          2. trit, scale = ternarize(W')（在旋转基下三值化）
          3. 存 trit (int8), scale (float16)
        """
        W = linear.weight.data.float()  # (out_features, in_features)
        assert W.shape == (self.out_features, self.in_features), (
            f"W.shape={W.shape} != (out={self.out_features}, in={self.in_features})"
        )

        # 1. Hadamard 旋转
        W_rot = self.rot.rotate_weight(W)

        # 2. 三值化 / 二值化
        if self.mode == "ternary":
            trit, scale = ternarize(W_rot, group_size=self.group_size)
        elif self.mode == "binary":
            trit, scale = binarize(W_rot, group_size=self.group_size)
        else:
            raise ValueError(f"mode 必须是 'ternary' 或 'binary'，收到 {self.mode}")

        self.trit.copy_(trit)
        # scale 形状随权重大小变化（num_groups = numel/group_size），动态赋值
        self.scale = scale.to(torch.float16)
        if linear.bias is not None and self.bias is not None:
            self.bias.copy_(linear.bias.data)
        self._prepared = True

    def forward(self, x: Tensor) -> Tensor:
        """前向：旋转 activation → 低 bit 乘 → 反旋转输出。

        数学：y = x @ W^T ≈ rotate_output( ternary_matmul( rotate_activation(x), trit, scale ) )
        """
        assert self._prepared, "先调 prepare_from_float() 烧权重"

        # 1. 旋转 activation 到匹配基
        x_rot = self.rot.rotate_activation(x)

        # 2. 低 bit matmul（权重不解压）
        if self.mode == "ternary":
            y_rot = ternary_linear(x_rot, self.trit, self.scale.float(), self.group_size)
        else:
            y_rot = binary_linear(x_rot, self.trit, self.scale.float(), self.group_size)

        # 3. 反旋转输出回原基
        y = self.rot.rotate_output(y_rot)

        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"mode={self.mode}, group={self.group_size}, block={self.block_size}"
        )