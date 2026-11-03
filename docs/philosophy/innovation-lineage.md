# 算法创新谱系：独创 vs 借鉴 vs 组合

> **主线**：Bonsai 的贡献不在发明单个技术，而在**系统整合** + **两个交叉创新**。
> 它站在 QuIP#、BitNet、Qwen3.6 的肩膀上，但站出了一个独有的姿态。
> 理清"哪些是独创、哪些是借鉴、哪些是组合"才能看清 Bonsai 的真正贡献在哪。

---

## 1. 为什么要理清创新谱系

### 1.1 评价一个系统的两种错误

评价 Bonsai 这样的系统，容易犯两种错误：

1. **过度归功**：把所有技术都说成"Bonsai 的创新"——不诚实，也掩盖了真正的贡献
2. **过度贬低**：发现某个技术来自上游论文，就结论"Bonsai 没创新"——忽略了系统整合本身的价值

正确的做法是**理清谱系**：哪些是独创、哪些是借鉴、哪些是组合，然后看**独创的部分是否有足够分量**。

### 1.2 Bonsai 的谱系分类

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

下面逐类讲清楚。

---

## 2. 真正的独创：DSpark block-denoising drafter

### 2.1 什么是 DSpark

**diffusion × speculative decoding 的交叉**——这是 Bonsai 最核心的算法独创。

传统 speculative decoding：小模型**逐个自回归**起草 token → 大模型逐个验证。
瓶颈：drafting 本身是串行的（草一个要看前一个），加速比受限于 drafting 速度。

DSpark 的创新：drafter 是 **block-parallel** 的，一次并行草 $k$ 个 token，
用 **diffusion-flavored block-denoising objective** 训练。

### 2.2 为什么这是"交叉"而不是"延伸"

- **从 speculative decoding 视角**：传统 drafter 是自回归的（串行），DSpark 改成 block-parallel（并行）。
  但"并行草 k 个 token"需要 drafter 能"不看前面就猜后面"——这不是 spec decoding 的自然延伸。

- **从 diffusion 视角**：diffusion 模型的去噪是"给定噪声版本，一次性预测全部干净 token"。
  DSpark 的 drafter 做的正是"给定 prefix，一次性预测 k 个后续 token"——
  把 diffusion 的"整块去噪"思路用在 token drafting 上。

两个领域的技术**交叉**，不是任一领域的直接延伸。这就是"独创"的含义。

### 2.3 关键设计细节

DSpark 的独创不只是"block-parallel"，还有三个训练设计：

1. **survival-probability-weighted distillation**：训练时给"更可能被接受的位置"更高权重。
   speculative decoding 的接受概率随位置递减（越后面越可能被拒），
   所以训练时给前面的位置高权重，让 drafter 在"最可能被接受的位置"上更准。

2. **per-source-normalized hidden-state taps**：从 target 的 5 个均匀间隔层 tap hidden state，做归一化后作为 drafter 输入。
   这让 drafter 能"看到" target 的中间表示，不只是最终 logits——信息更丰富。

3. **draft block size $k$ 从 measured verify-cost model 选**：先测验证成本，再反推最优 $k$。
   验证成本 = target 验证 $k$ 个 token 的时间。drafting 成本 = drafter 1 次前向的时间。
   最优 $k$ = 让 (drafting + verify) 最小的 $k$。

### 2.4 反直觉的 workload × 硬件依赖

| 场景 | L40S (CUDA) | M5 Max (Metal) |
|---|---|---|
| ternary code/math | 1.8–2.4x | ~1.2x |
| ternary chat/reasoning | 较少 | **更慢** |
| 1-bit 任何场景 | 1.4–1.75x | **更慢** |

**code/math 收益最大**：下一个 token 高度可预测（语法结构强），drafter 接受率高。
block-denoising 的"看不到同块后面"的劣势小——高可预测性意味着 $t_{i+1}$ 基本由 prefix 决定。

**chat/reasoning 收益少**：自然语言多样性高，$t_{i+1}$ 强依赖 $t_i$，看不到就猜不准。

**Apple Silicon 上更慢**：batch-1 的验证 pass 在 Metal 上不 amortize。
这不是"还没优化"，是**架构层面的根本性差异**。

### 2.5 代价

- **跨请求 prompt-cache 复用被禁**：每个请求要 re-prefill 全部历史
- **单 slot**（`-np 1`）：不能并行处理多个请求

这是"用吞吐换延迟"的选择——适合交互式端侧场景，不适合高并发服务端。

### 2.6 上游化

DSpark 已 upstream 到 mainline llama.cpp（PR #25173，`draft-dspark`）。
标志着 Bonsai 的算法创新已被社区认可，成为通用基础设施。

---

## 3. 借鉴 + 彻底组合：三值化 + 旋转 + group scale

### 3.1 三个技术的上游来源

| 技术 | 上游来源 | 上游做了什么 | Bonsai 的贡献 |
|---|---|---|---|
| 三值化 $\{−1,0,+1\}$ | BitNet 1.58b (Ma et al.) | 证明三值化可行，需 QAT | 配合旋转 + group scale 端到端 |
| Hadamard-domain quant. | QuIP# (Chee et al.) | 用旋转 + 2-bit codebook | 改用三值化，折叠进存储 |
| Group scale (g128) | 通用量化技术 | per-group scale 提精度 | FP16 scale，0.125 bpw |

### 3.2 单独看都不够

- **QuIP#**：用旋转但没做三值化（用 2-bit codebook）——codebook 需要 2 bit，不够激进
- **BitNet**：用三值化但没做旋转（依赖 QAT 硬扛）——没有旋转，误差沿敏感方向集中
- **通用 group scale**：只是精度工具，不解决"塌陷"问题

### 3.3 Bonsai 的组合：乘法不是加法

Bonsai 把三者**乘起来**：

- 旋转改变误差方向 → 误差均匀分布
- 三值化在旋转后的好基上做 → 误差均匀且小
- QAT 让权重适应"旋转 + 三值化"的联合约束 → 权重主动配合
- group scale 保留每组的幅度信息 → 补回三值化丢失的量级

这个组合不是"加法"（各自贡献叠加），而是"乘法"（互相放大）：
旋转让三值化的误差均匀化，QAT 让权重在旋转后的基上学会用三值表达信息，
group scale 补回三值化丢的幅度。

### 3.4 "折叠进存储"的工程创新

QuIP# 的旋转是"推理时做"，有额外计算开销。
Bonsai 把旋转**折叠进存储权重**（离线做一次 $W' = H \cdot W \cdot H^\top$），
运行时只需对 activation 做匹配变换。

**不花额外存储 bit，不花额外权重带宽。** 这是一个工程层面的创新——
不是新算法，但让旋转从"有开销的优化"变成"零开销的正确性条件"。

---

## 4. 纯借鉴：Hybrid Attention

75% linear / 25% full 的 hybrid attention 直接来自 Qwen3.6 架构。
Bonsai 没有改进这个技术，只是继承了它。

这不算是 Bonsai 的贡献，但它是系统的一部分——
linear attention 的 $O(n)$ 复杂度在低 bit 约束下最大化有效上下文长度。

RoPE、RMSNorm、SwiGLU 也都是 Qwen/Llama 系的标配，Bonsai 直接继承。

---

## 5. 度量层面的贡献：Intelligence Density

$$D = \frac{-\log_2(1 - \text{score}/100)}{\text{size\_GB}}$$

这个度量本身是一个贡献——它重新定义了端侧 AI 的优化目标：
从"固定大小最大化精度"转向"固定'装得下'约束最大化密度"。

详见 [Intelligence Density](intelligence-density.md)。

有人可能说"这不就是 accuracy/size 吗？"——不是。
$-\log_2$ 的非线性正确刻画了"可用性"的非线性：
57 分的模型不可用（密度低），85 分的模型可用（密度高），
线性度量会给 57 分的模型不合理的非零密度。

---

## 6. "无 escape hatch" 的系统设计

### 6.1 什么是无 escape hatch

Bonsai 的一个关键设计哲学是**无 escape hatch**：

- 不允许"重要层保留高 bit"（混合精度）——所有层都是三值/二值
- 不允许"难量化的权重跳过旋转"——全部旋转
- 不允许"推理时解压回 FP16"——权重以打包格式直接被 kernel 消费

### 6.2 为什么这很重要

这种"无退路"的设计**倒逼系统在每个环节都做到极致**。
如果有 escape hatch，工程上会倾向于"难的地方就跳过"，
最终模型变成"半量化半 FP16"的缝合怪，既不省空间也不保精度。

!!! insight "约束催生创新"
    "无 escape hatch"不是限制，而是**设计约束**。
    正是因为"必须全部三值化"的硬约束，才倒逼出"旋转 + QAT + group scale"的系统配合。
    如果允许混合精度，这个系统配合可能永远不会被发明。

    这是"约束催生创新"的经典案例——和 C 语言的手动内存管理催生 Rust 的所有权系统同理。

---

## 7. 总结：Bonsai 的真正贡献

| 贡献类型 | 具体内容 | 分量 |
|---|---|---|
| **独创** | DSpark block-denoising drafter | 重——新算法，已上游化 |
| **独创** | Intelligence Density 度量 | 中——新视角，影响选型 |
| **组合** | 旋转 × 三值化 × QAT × group scale | 重——乘法关系，缺一不可 |
| **工程** | 旋转折叠进存储（零额外 bit） | 中——让旋转零开销 |
| **借鉴** | hybrid attention, RoPE, RMSNorm | 轻——直接继承 |

**核心结论**：Bonsai 的贡献不在单个技术，而在**系统整合** + **两个交叉创新**（DSpark、Intelligence Density）。
它站在 QuIP#、BitNet、Qwen3.6 的肩膀上，但站出了一个独有的姿态。

---

## 进一步阅读

- [DSpark drafter](../operators/dspark.md) — 独创的 block-denoising drafter 详解
- [Intelligence Density](intelligence-density.md) — 度量层面的贡献
- [为什么 1.7 bit 不崩](why-not-collapse.md) — 系统整合的效果