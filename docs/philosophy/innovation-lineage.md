# 算法创新谱系：独创 vs 借鉴 vs 组合

> **核心见解**：Bonsai 的贡献不是发明单个技术，而是**把已有技术端到端整合成一个无 escape hatch 的系统**，
> 并在两个交叉点上做出了真正的独创。

## 创新谱系分类

### 1. 真正的独创：DSpark block-denoising drafter

**diffusion × speculative decoding 的交叉**——这是 Bonsai 最核心的算法独创。

传统 speculative decoding：小模型**逐个自回归**起草 token → 大模型逐个验证。
瓶颈：drafting 本身是串行的（草一个要看前一个），加速比受限于 drafting 速度。

DSpark 的创新：drafter 是 **block-parallel** 的，一次并行草 $k$ 个 token，
用 **diffusion-flavored block-denoising objective** 训练。

这是把 diffusion 模型的"整块去噪"思路嫁接到 token drafting——
两个领域的技术交叉，不是任一领域的直接延伸。

**关键设计细节**：

- **survival-probability-weighted distillation**：训练时给"更可能被接受的位置"更高权重
- **per-source-normalized hidden-state taps**：从 target 5 个均匀间隔层 tap hidden state
- **draft block size $k$** 从 **measured verify-cost model** 选——先测验证成本，再反推最优 $k$

**反直觉的 workload × 硬件依赖**：

| 场景 | L40S (CUDA) | M5 Max (Metal) |
|---|---|---|
| ternary code/math | 1.8–2.4x | ~1.2x |
| ternary chat/reasoning | 较少 | **更慢** |
| 1-bit 任何场景 | 1.4–1.75x | **更慢** |

Apple Silicon 上更慢——batch-1 的验证 pass 在 Metal 上不 amortize。
这不是"还没优化"，是**架构层面的根本性差异**。

### 2. 借鉴 + 彻底组合：三值化 + 旋转 + group scale

| 技术 | 上游来源 | Bonsai 的贡献 |
|---|---|---|
| 三值化 $\{−1,0,+1\}$ | BitNet 1.58b (Ma et al.) | 配合旋转 + QAT 端到端 |
| Hadamard-domain quantization | QuIP# (Chee et al.) | 折叠进存储，零额外 bit |
| Group scale (g128) | 通用量化技术 | FP16 scale，0.125 bpw 开销 |

单独看，这三个都是已有技术。Bonsai 的贡献在**组合方式**：

- QuIP# 用旋转但没做三值化（用 2-bit codebook）
- BitNet 用三值化但没做旋转（依赖 QAT 硬扛）
- Bonsai 把两者**乘起来**：旋转改变误差方向 + 三值化在好基上做 + QAT 让权重适应

这个组合不是"加法"（各自贡献叠加），而是"乘法"（互相放大）：
旋转让三值化的误差均匀化，QAT 让权重在旋转后的基上学会用三值表达信息。

### 3. 纯借鉴：Hybrid Attention

75% linear / 25% full 的 hybrid attention 直接来自 Qwen3.6 架构。
Bonsai 没有改进这个技术，只是继承了它。

这不算是 Bonsai 的贡献，但它是系统的一部分——
linear attention 的 $O(n)$ 复杂度在低 bit 约束下最大化有效上下文长度。

### 4. 度量层面的贡献：Intelligence Density

$$D = \frac{-\log_2(1 - \text{score}/100)}{\text{size\_GB}}$$

这个度量本身是一个贡献——它重新定义了端侧 AI 的优化目标：
从"固定大小最大化精度"转向"固定'装得下'约束最大化密度"。

详见 [Intelligence Density](intelligence-density.md)。

## 创新谱系总览

```
                    ┌─ 独创 ─────────────────────────────────┐
                    │  DSpark block-denoising drafter        │
                    │  (diffusion × spec decoding 交叉)      │
                    │  Intelligence Density 度量             │
                    └────────────────────────────────────────┘
                    ┌─ 借鉴 + 彻底组合 ──────────────────────┐
Bonsai-27B  =       │  三值化 ← BitNet 1.58b                 │
                    │  Hadamard 旋转 ← QuIP#                 │
                    │  group scale ← 通用量化                │
                    │  (组合是乘法，不是加法)                │
                    └────────────────────────────────────────┘
                    ┌─ 纯借鉴 ───────────────────────────────┐
                    │  hybrid attention ← Qwen3.6            │
                    │  RoPE, RMSNorm, SwiGLU ← Qwen/Llama 系 │
                    └────────────────────────────────────────┘
```

## "无 escape hatch" 的系统设计

Bonsai 的一个关键设计哲学是**无 escape hatch**：

- 不允许"重要层保留高 bit"（混合精度）——所有层都是三值/二值
- 不允许"难量化的权重跳过旋转"——全部旋转
- 不允许"推理时解压回 FP16"——权重以打包格式直接被 kernel 消费

这种"无退路"的设计**倒逼系统在每个环节都做到极致**。
如果有 escape hatch，工程上会倾向于"难的地方就跳过"，最终模型变成"半量化半 FP16"的缝合怪，
既不省空间也不保精度。

!!! insight "见解：约束催生创新"
    Bonsai 的"无 escape hatch"不是限制，而是**设计约束**。
    正是因为"必须全部三值化"的硬约束，才倒逼出"旋转 + QAT + group scale"的系统配合。
    如果允许混合精度，这个系统配合可能永远不会被发明。

    这是"约束催生创新"的经典案例——和 C 语言的手动内存管理催生 Rust 的所有权系统同理。

## 上游化

DSpark 已 upstream 到 mainline llama.cpp（PR #25173，`draft-dspark`）。
这标志着 Bonsai 的算法创新已被社区认可，成为通用基础设施。

---

!!! quote "一句话总结"
    Bonsai 的贡献不在单个技术，而在**系统整合** + **两个交叉创新**（DSpark、Intelligence Density）。
    它站在 QuIP#、BitNet、Qwen3.6 的肩膀上，但站出了一个独有的姿态。