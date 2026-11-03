"""DSpark block-denoising drafter —— Bonsai 独创的 diffusion × speculative decoding。

传统 speculative decoding：小模型逐个自回归起草 → 大模型逐个验证。瓶颈：drafting 串行。
DSpark：drafter block-parallel 一次草 k 个 token，用 diffusion-flavored block-denoising objective 训练。

关键设计：
  - survival-probability-weighted distillation：给"更可能被接受的位置"高权重
  - per-source-normalized hidden-state taps（从 target 5 层 tap）
  - draft block size k 从 measured verify-cost model 选

反直觉的 workload × 硬件依赖：
  L40S CUDA: ternary 1.8-2.4x (code/math), 1.4-1.75x (1-bit)
  M5 Max Metal: code/math ~1.2x, chat/reasoning 更慢 → 不推荐
  → 收益强 workload-dependent × 硬件-dependent

代价：跨请求 prompt-cache 复用被禁，单 slot (-np 1)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TargetModel(nn.Module):
    """大模型（target）：教学用 embedding + GRU + linear，够提供"真实"的 next-token 分布。"""

    def __init__(self, vocab: int = 100, dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, vocab)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embed(tokens.unsqueeze(0))
        out, _ = self.rnn(x)
        return self.head(out).squeeze(0)


class AutoregressiveDrafter(nn.Module):
    """传统自回归 drafter：逐个草 token，每个依赖前一个。串行 k 次前向。"""

    def __init__(self, vocab: int = 100, dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, vocab)

    def draft(self, prefix: Tensor, k: int) -> Tensor:
        drafted = []
        cur = prefix.clone()
        for _ in range(k):
            x = self.embed(cur.unsqueeze(0))
            out, _ = self.rnn(x)
            logits = self.head(out[:, -1])
            tok = logits.argmax(dim=-1)
            drafted.append(tok.item())
            cur = torch.cat([cur, tok])
        return torch.tensor(drafted)


class BlockDenoisingDrafter(nn.Module):
    """DSpark 风格 block-parallel drafter：一次并行草 k 个 token。

    创新：不逐个草，给定 prefix 一次性预测整块 k 个 token。
    用 block-denoising 训练。速度：1 次前向 vs 自回归的 k 次。
    质量：可能比自回归差（每个 token 看不到同块后面的），但验证保无损。
    """

    def __init__(self, vocab: int = 100, dim: int = 32, block_size: int = 4):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, vocab * block_size)

    def draft(self, prefix: Tensor, k: int) -> Tensor:
        assert k == self.block_size
        x = self.embed(prefix.unsqueeze(0))
        out, _ = self.rnn(x)
        logits = self.head(out[:, -1])
        logits = logits.view(k, -1)
        return logits.argmax(dim=-1)


def verify(target: nn.Module, prefix: Tensor, drafted: Tensor) -> int:
    """验证：target 逐个检查 drafted token 是否匹配自己的 argmax。

    返回接受数（从第一个不匹配位置截断）。
    验证保持目标分布 → 无损：accepted token 和普通生成不可区分。
    """
    accepted = 0
    cur = prefix.clone()
    for tok in drafted:
        x = target.embed(cur.unsqueeze(0))
        out, _ = target.rnn(x)
        target_tok = target.head(out[:, -1]).argmax(dim=-1).item()
        if tok.item() == target_tok:
            accepted += 1
            cur = torch.cat([cur, torch.tensor([tok.item()])])
        else:
            break
    return accepted


def train_drafter(
    drafter: nn.Module,
    target: nn.Module,
    data: Tensor,
    steps: int = 200,
) -> None:
    """训练 drafter 预测真实序列 data 的后续 token。

    block-denoising drafter：一次预测 k 个真实 token（block-denoising objective）。
    自回归 drafter：逐个预测下一个真实 token。
    ground truth 用 data 本身（target 也学 data，两者一致 → 接受率高）。
    """
    opt = torch.optim.Adam(drafter.parameters(), lr=1e-2)
    target.eval()
    k = drafter.block_size if isinstance(drafter, BlockDenoisingDrafter) else 1
    for _ in range(steps):
        i = torch.randint(2, len(data) - k - 1, (1,)).item()
        prefix = data[:i]
        target_k = data[i:i + k]

        x = drafter.embed(prefix.unsqueeze(0))
        out, _ = drafter.rnn(x)
        if isinstance(drafter, BlockDenoisingDrafter):
            logits = drafter.head(out[:, -1]).view(k, -1)
        else:
            logits = drafter.head(out[:, -1])
        loss = F.cross_entropy(logits, target_k)

        opt.zero_grad()
        loss.backward()
        opt.step()


def make_sequence(kind: str, length: int = 200, vocab: int = 100) -> Tensor:
    """造两种序列：code-like（高可预测）和 chat-like（低可预测）。

    code-like：重复结构 + 强 pattern（模拟代码的语法约束）
      → 下一个 token 高度可预测，drafter 接受率高
    chat-like：随机 + 少量 pattern（模拟自然语言的多样性）
      → 下一个 token 不确定性强，drafter 接受率低
    """
    torch.manual_seed(42)
    if kind == "code":
        pattern = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        seq = pattern.repeat(length // 10 + 1)[:length]
        noise_idx = torch.randperm(length)[:length // 20]
        seq[noise_idx] = torch.randint(0, vocab, (len(noise_idx),))
        return seq
    else:
        seq = torch.randint(0, vocab, (length,))
        n_pattern = length // 10
        idx = torch.randperm(length)[:n_pattern]
        seq[idx] = torch.tensor([1, 2, 3, 4, 5]).repeat(n_pattern // 5 + 1)[:n_pattern]
        return seq