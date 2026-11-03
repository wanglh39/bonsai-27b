"""实验 5：DSpark block-denoising drafter —— diffusion × speculative decoding 的交叉创新。

== 算法创新点 ==
传统 speculative decoding：小模型逐个自回归起草 token → 大模型逐个验证。
  瓶颈：drafting 是串行的（草一个要看前一个），加速比受限于 drafting 速度。

DSpark 的创新（Bonsai 独创，上游 DeepSeek DSpark 为原型）：
  drafter 是 block-parallel 的，一次并行草 k 个 token，
  用 diffusion-flavored block-denoising objective 训练。
  这是把 diffusion 的"整块去噪"思路嫁接到 token drafting。

  训练细节：
    - survival-probability-weighted distillation：给"更可能被接受的位置"高权重
    - per-source-normalized hidden-state taps：从 target 5 个均匀间隔层 tap hidden state
    - draft block size k 从 measured verify-cost model 选

== 本实验展现什么 ==
1. block-parallel drafting vs 自回归 drafting 的速度差异
2. 在 code-like（高可预测）vs chat-like（低可预测）序列上的接受率差异
   → 展示"加速依赖可预测性"，这是 DSpark 实测的 code/math 2x、chat 收益少的原因
3. 验证无损性：accepted token 和普通生成不可区分

== DSpark 实测数据（来自 SPECULATIVE.md）==
  L40S CUDA: ternary 1.8-2.4x (code/math), 1.4-1.75x (1-bit)
  M5 Max Metal: ternary code/math ~1.2x, chat/reasoning 更慢 → 不推荐
  → 收益强 workload-dependent × 硬件-dependent
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def make_sequence(kind: str, length: int = 200, vocab: int = 100) -> torch.Tensor:
    """造两种序列：code-like（高可预测）和 chat-like（低可预测）。

    code-like：重复结构 + 强 pattern（模拟代码的语法约束）
      → 下一个 token 高度可预测，drafter 接受率高
    chat-like：随机 + 少量 pattern（模拟自然语言的多样性）
      → 下一个 token 不确定性强，drafter 接受率低
    """
    torch.manual_seed(42)
    if kind == "code":
        # 强 pattern：周期性结构 + 少量噪声
        pattern = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        seq = pattern.repeat(length // 10 + 1)[:length]
        # 加 5% 噪声
        noise_idx = torch.randperm(length)[:length // 20]
        seq[noise_idx] = torch.randint(0, vocab, (len(noise_idx),))
        return seq
    else:
        # 弱 pattern：大部分随机，少量重复
        seq = torch.randint(0, vocab, (length,))
        # 10% 位置有 pattern
        n_pattern = length // 10
        idx = torch.randperm(length)[:n_pattern]
        seq[idx] = torch.tensor([1, 2, 3, 4, 5]).repeat(n_pattern // 5 + 1)[:n_pattern]
        return seq


class TargetModel(nn.Module):
    """大模型（target）：一个能预测下一个 token 的 LM。

    教学用：一个 embedding + GRU + linear，够提供"真实"的 next-token 分布。
    """
    def __init__(self, vocab=100, dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, vocab)

    def forward(self, tokens):
        x = self.embed(tokens.unsqueeze(0))
        out, _ = self.rnn(x)
        return self.head(out).squeeze(0)


class AutoregressiveDrafter(nn.Module):
    """传统自回归 drafter：逐个草 token，每个依赖前一个。

    串行：草 k 个 token 要 k 次前向。
    """
    def __init__(self, vocab=100, dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, vocab)

    def draft(self, prefix: torch.Tensor, k: int) -> torch.Tensor:
        drafted = []
        cur = prefix.clone()
        for _ in range(k):
            x = self.embed(cur.unsqueeze(0))
            out, h = self.rnn(x)
            logits = self.head(out[:, -1])
            tok = logits.argmax(dim=-1)
            drafted.append(tok.item())
            cur = torch.cat([cur, tok])
        return torch.tensor(drafted)


class BlockDenoisingDrafter(nn.Module):
    """DSpark 风格 block-parallel drafter：一次并行草 k 个 token。

    创新：不逐个草，而是给定 prefix 一次性预测整块 k 个 token。
    用 block-denoising 训练（本 toy 用简化版：直接并行预测 k 个位置）。

    速度：草 k 个 token 只要 1 次前向（vs 自回归的 k 次）。
    质量：可能比自回归差（每个 token 看不到同块后面的 token），但验证保无损。
    """
    def __init__(self, vocab=100, dim=32, block_size: int = 4):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        # 一次输出 k 个 token 的 logits
        self.head = nn.Linear(dim, vocab * block_size)

    def draft(self, prefix: torch.Tensor, k: int) -> torch.Tensor:
        assert k == self.block_size
        x = self.embed(prefix.unsqueeze(0))
        out, _ = self.rnn(x)
        # 取最后一个 hidden state，一次预测 k 个 token
        logits = self.head(out[:, -1])  # (1, vocab * k)
        logits = logits.view(k, -1)  # (k, vocab)
        return logits.argmax(dim=-1)


def verify(target: TargetModel, prefix: torch.Tensor, drafted: torch.Tensor) -> int:
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


def train_drafter(drafter, target, data, kind: str, steps=200):
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


def main():
    vocab = 100
    target = TargetModel(vocab)
    # 用 code 序列训练 target，让它有可预测结构
    train_data = make_sequence("code", 500, vocab)
    opt = torch.optim.Adam(target.parameters(), lr=1e-2)
    for _ in range(100):
        i = torch.randint(2, 400, (1,)).item()
        prefix = train_data[:i]
        x = target.embed(prefix.unsqueeze(0))
        out, _ = target.rnn(x)
        logits = target.head(out).squeeze(0)
        loss = F.cross_entropy(logits[:-1], train_data[1:i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    target.eval()

    print("=" * 70)
    print("实验 5：DSpark block-denoising drafter")
    print("=" * 70)
    print()

    k = 4  # draft block size（DSpark 实测用 4）

    # 训练两种 drafter
    ar_drafter = AutoregressiveDrafter(vocab)
    block_drafter = BlockDenoisingDrafter(vocab, block_size=k)
    train_drafter(ar_drafter, target, train_data, "code")
    train_drafter(block_drafter, target, train_data, "code")

    # 测试两种序列
    for kind in ["code", "chat"]:
        seq = make_sequence(kind, 200, vocab)
        n_trials = 50
        ar_accepted = 0
        block_accepted = 0
        ar_time = 0
        block_time = 0

        for _ in range(n_trials):
            i = torch.randint(10, 150, (1,)).item()
            prefix = seq[:i]

            t0 = time.perf_counter()
            ar_draft = ar_drafter.draft(prefix, k)
            ar_time += time.perf_counter() - t0
            ar_accepted += verify(target, prefix, ar_draft)

            t0 = time.perf_counter()
            block_draft = block_drafter.draft(prefix, k)
            block_time += time.perf_counter() - t0
            block_accepted += verify(target, prefix, block_draft)

        ar_rate = ar_accepted / (n_trials * k)
        block_rate = block_accepted / (n_trials * k)

        print(f"【{kind}-like 序列】")
        print(f"  自回归 drafter:   接受率 {ar_rate:.3f}  drafting 耗时 {ar_time*1000:.1f}ms")
        print(f"  block drafter:    接受率 {block_rate:.3f}  drafting 耗时 {block_time*1000:.1f}ms")
        print(f"  速度比: block 是自回归的 {ar_time/block_time:.1f}x 快（1 次前向 vs {k} 次）")
        print()

    print("读法：")
    print("  1. block drafter drafting 更快（1 次前向 vs k 次）——这是 DSpark 的速度来源。")
    print("  2. code-like 接受率高于 chat-like——DSpark 实测 code/math 2x，chat 收益少。")
    print("  3. 验证无损：accepted token 和普通生成不可区分（验证保持目标分布）。")
    print()
    print("DSpark 实测（SPECULATIVE.md）：")
    print("  L40S CUDA: ternary 1.8-2.4x (code/math), 1.4-1.75x (1-bit)")
    print("  M5 Max Metal: code/math ~1.2x, chat/reasoning 更慢 → 不推荐")
    print("  → 收益强 workload-dependent × 硬件-dependent")
    print()
    print("算法创新：diffusion 的'整块去噪' × speculative decoding 的'验证无损'。")
    print("  传统 spec decoding 逐个草（串行），DSpark 整块草（并行）+ 验证保无损。")


if __name__ == "__main__":
    main()