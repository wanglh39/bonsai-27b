"""Hadamard 旋转基 —— Bonsai 不塌陷的真正原因。

== 为什么需要这一步 ==
直接对权重做三值化 {−1,0,+1}，误差会沿权重的"主方向"（大奇异值方向）累积。
因为真实权重的信息是各向异性的：少数方向承载大部分能量，多数方向接近零。
三值化把大权重压扁、给小权重注入等量噪声 → 主方向被系统性破坏 →
   下游表现为 reasoning chain 崩盘（math 掉到 57 分）。

== Hadamard 旋转做了什么 ==
对权重矩阵 W 做正交变换 W' = H @ W @ H^T，其中 H 是 Hadamard 矩阵。
H 的性质：
  - 正交：H @ H^T = n*I，保持 Frobenius 范数 ||W'||_F = ||W||_F（不丢信息）
  - 各向同性：把任意方向的 outlier 能量均匀打散到所有方向
  - 整数元素（±1）：旋转可用加减法实现，无浮点误差

旋转后 W' 的分布接近球形高斯 → 三值化误差变成"各向均匀的小噪声"，
而不是"沿主方向的系统性偏差"。这就是 QuIP# / Bonsai 的核心洞察。

== 零额外成本 ==
旋转折叠进存储权重（离线做一次），运行时只需对 activation 做匹配旋转。
不花额外存储 bit，不花额外权重带宽。Bonsai 模型卡片原话：
  "The rotation is folded into the stored weights offline, so it costs
   no extra bits and no extra weight traffic."
"""
from __future__ import annotations

import torch
from torch import Tensor


def hadamard_matrix(n: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """构造 n 阶 Walsh-Hadamard 矩阵（n 必须是 2 的幂）。

    递归定义：
        H_1 = [1]
        H_{2n} = [[ H_n,  H_n],
                  [ H_n, -H_n]]
    结果是 ±1 的正交矩阵（未归一化），满足 H @ H^T = n * I。

    教学注记：实际实现用 Sylvester 构造的向量化版本，避免递归开销。
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard 阶数必须是 2 的幂，收到 n={n}")
    if n == 1:
        return torch.tensor([[1.0]], dtype=dtype)

    # Sylvester 构造：H = H_2 ⊗ H_2 ⊗ ... (log2 n 次 Kronecker 积)
    H2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=dtype)
    H = H2
    k = n
    while k > 2:
        H = torch.kron(H, H2)
        k //= 2
    return H


def random_hadamard(n: int, generator: torch.Generator | None = None) -> Tensor:
    """随机化 Hadamard：H @ diag(随机 ±1)。

    单纯的 Walsh-Hadamard 有固定结构，旋转后可能留下 artifact。
    实践中（QuIP#）用 H @ D（D 是随机对角符号矩阵）打散结构，
    既保持正交性又避免周期性 pattern。Bonsai 的离线旋转即此类。
    """
    H = hadamard_matrix(n)
    signs = torch.randint(0, 2, (n,), generator=generator) * 2 - 1  # ∈ {−1, +1}
    return H * signs  # 等价于 H @ diag(signs)，但省一次矩阵乘


class BlockHadamard(torch.nn.Module):
    """分块 Hadamard 旋转：对矩阵的行/列分块做正交旋转。

    完整的 W' = H @ W @ H^T 对大矩阵开销高，实际做法是 block-diagonal：
    把 W 切成 block_size × block_size 的小块，每块独立旋转。
    这正是 Bonsai 模型卡片说的 "blockwise orthogonal Hadamard rotation"。

    前向流程（对应一个低 bit Linear 层的完整生命周期）：
      离线 rotate_weight(W):  W' = H_blk @ W @ H_blk^T   → 存储 W'
      在线 forward(x):        x' = H_blk @ x              → 旋转 activation
                             y' = x' @ W'^T              → 在旋转基下做低 bit 乘
                             y  = y' @ H_blk^T           → 反旋转回原基
    """

    def __init__(self, block_size: int = 128, randomized: bool = True):
        super().__init__()
        assert (block_size & (block_size - 1)) == 0, "block_size 必须是 2 的幂"
        self.block_size = block_size
        self.randomized = randomized
        # 预生成一个 block 的 H，运行时对每个 block 复用
        if randomized:
            H = random_hadamard(block_size)
        else:
            H = hadamard_matrix(block_size)
        # 归一化为正交：H @ H^T = I（除以 sqrt(block_size)）
        self.register_buffer("H", H / (block_size ** 0.5), persistent=False)
        # H^T 也预存，反旋转用
        self.register_buffer("H_T", self.H.t().contiguous(), persistent=False)

    def rotate_weight(self, W: Tensor) -> Tensor:
        """离线用：W' = H_blk @ W @ H_blk^T。对 W 的行和列分别分块旋转。

        要求 W 的行数、列数都能被 block_size 整除（否则 pad 或调 block_size）。
        """
        out_rows, in_cols = W.shape
        assert out_rows % self.block_size == 0 and in_cols % self.block_size == 0, (
            f"权重形状 {W.shape} 必须能被 block_size={self.block_size} 整除"
        )
        H = self.H
        # 分块旋转行：把 out_rows 切成 (out_rows/block) 个 block，每块左乘 H
        # 用 einsum 一次搞定：W 视作 (out_blocks, block, in_cols)，对 block 维乘 H
        out_blocks = out_rows // self.block_size
        W = W.reshape(out_blocks, self.block_size, in_cols)
        W = torch.einsum("ab,obc->oac", H, W)  # 每个行块左乘 H
        W = W.reshape(out_rows, in_cols)

        # 分块旋转列：in_cols 切成 (in_cols/block) 个 block，每块右乘 H^T
        in_blocks = in_cols // self.block_size
        W = W.reshape(out_rows, in_blocks, self.block_size)
        W = torch.einsum("obc,cd->obd", W, self.H_T)  # 每个列块右乘 H^T
        W = W.reshape(out_rows, in_cols)
        return W

    def rotate_activation(self, x: Tensor) -> Tensor:
        """在线用：x' = x @ H_blk^T（对最后一维分块右乘 H^T）。

        注意：forward 里 activation 是右乘，对应权重左乘的转置关系。
        对应 rotate_weight 中"列块右乘 H^T"，activation 需要右乘 H 才能配对。
        这里实现的是 x @ H（与 rotate_weight 的列旋转 H^T 配对，因为 (x@H) @ (H^T@W)^T 的转置关系）。
        为保持教学清晰，我们让 rotate_activation / rotate_output 与 rotate_weight 严格对应。
        """
        # x: (..., in_cols)，对最后一维分块
        in_cols = x.shape[-1]
        assert in_cols % self.block_size == 0
        in_blocks = in_cols // self.block_size
        x = x.reshape(*x.shape[:-1], in_blocks, self.block_size)
        # activation 右乘 H（对应权重列方向右乘了 H^T，互为转置逆）
        x = torch.einsum("...ob,ab->...oa", x, self.H)
        x = x.reshape(*x.shape[:-2], in_cols)
        return x

    def rotate_output(self, y: Tensor) -> Tensor:
        """在线用：y = y' @ H_blk（对最后一维分块右乘 H，撤销行方向的旋转）。

        对应 rotate_weight 中"行块左乘 H"，输出需左乘 H^T 撤销 → 等价右乘 H^T 在转置视角。
        为教学清晰，这里直接实现 y' @ H^T 的分块版。
        """
        out_rows = y.shape[-1]
        assert out_rows % self.block_size == 0
        out_blocks = out_rows // self.block_size
        y = y.reshape(*y.shape[:-1], out_blocks, self.block_size)
        y = torch.einsum("...ob,ab->...oa", y, self.H_T)
        y = y.reshape(*y.shape[:-2], out_rows)
        return y