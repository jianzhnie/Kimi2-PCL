# PCl-Model-1T 大语言模型评测报告

## 1. 评测概览

本报告记录了 PCL-Model-1T 大语言模型在预训练阶段的关键性能指标。

- **模型阶段**: 基于预训练 10,000 Step 保存的 Checkpoint。
- **训练规模**: 已完成约 4.0T Tokens 的训练。
- **评测环境**: 采用 `vLLM` 推理引擎配合 `lmeval` 框架。
- **硬件平台**: 昇腾 (Ascend) NPU 集群。

---

## 2. 核心基准测试结果

### 2.1 MMLU (Massive Multitask Language Understanding)

MMLU 是衡量语言模型综合理解能力的核心基准，涵盖了从基础教育到专业水平的 57 个主题。评测结果分为四个大类：**人文 (Humanities)**、**社会科学 (Social Sciences)**、**STEM (科学、技术、工程、数学)** 以及 **其他 (Other)**。

#### 2.1.1 Zero-shot 结果 (n-shot: 0)

| 任务 (Task) | 准确率 (acc) ↑ | 标准误差 (acc_stderr) |
| :--- | :---: | :---: |
| **MMLU (Overall)** | **0.3441** | 0.0040 |
| - Humanities | 0.3092 | 0.0067 |
| - Social Sciences | 0.3776 | 0.0087 |
| - STEM | 0.3283 | 0.0083 |
| - Other | 0.3798 | 0.0087 |

#### 2.1.2 Few-shot 结果 (n-shot: 5)

| 任务 (Task) | 准确率 (acc) ↑ | 标准误差 (acc_stderr) |
| :--- | :---: | :---: |
| **MMLU (Overall)** | **0.3506** | 0.0040 |
| - Humanities | 0.3184 | 0.0067 |
| - Social Sciences | 0.3760 | 0.0087 |
| - STEM | 0.3279 | 0.0083 |
| - Other | 0.3972 | 0.0087 |

> **数据说明**：
> - **Task**: `mmlu` 为全量 57 个子任务的宏平均结果；各子分类（如 Humanities）为其下属主题的加权平均。
> - **准确率 (acc)**: 模型预测正确的样本比例，已四舍五入保留 4 位小数。
> - **标准误差 (acc_stderr)**: 反映评测结果的统计稳定性，数值越小表示结果越可靠。

---

### 2.2 Wikitext-2 Perplexity (语言建模能力)

Wikitext-2 数据集用于评估模型对高质量维基百科文本的语言建模能力。**困惑度 (Perplexity, PPL)** 是衡量模型预测文本序列能力的关键指标。

#### Wikitext-2 评测结果

| 评测指标 (Metric) | 结果 (Value) ↓ | 说明 |
| :--- | :---: | :--- |
| **Word Perplexity** | **15.7494** | 基于单词层面的预测困惑度 |
| Byte Perplexity | 1.6745 | 基于字节层面的预测困惑度 |
| Bits per Byte (BPB) | 0.7438 | 平均每个字节所需的比特数 |


## 3. 推理评测适配过程介绍

PCL-1T-Model 项目基于开源的 Kimi-K2-Base (`moonshotai/Kimi-K2-Base`) 架构进行了深度改造：将注意力机制从 **MLA (Multi-head Latent Attention)** 替换为 **GQA (Grouped Query Attention)**，引入 QK LayerNorm，并大幅调整了 MoE 专家规模与路由策略。由于架构差异，推理评测需要完成一系列适配工作。

以下按 **模型架构理解 → 权重转换 → 推理框架适配 → 评测执行** 的顺序，逐步介绍完整流程。

---

### 3.1 模型架构对比

> **核心设计思路**: PCL-1T-Model 虽然层数从 61 减至 32，但将 MoE 单专家维度从 2048 提升到 12288（6 倍），在 128 个路由专家下支撑起约 **1T** 的参数规模。同时从 MLA 切换到 GQA（`num_query_groups=2`），降低了 KV Cache 开销。

#### 3.1.1 详细参数对比

| 参数模块 | 参数名称 | Kimi-K2-Base | PCL-Model (1T) | 说明 |
|:--|:--|:--|:--|:--|
| **基础规模** | `num_hidden_layers` | 61 | 32 | 层数减半 |
| | `hidden_size` | 7168 | 7168 | 不变 |
| | `vocab_size` | 163840 | 163840 | 不变 |
| | `max_position_embeddings` | 131072 | 131072 | 不变 |
| **注意力机制** | `num_attention_heads` | 64 | 64 | 不变 |
| | `num_key_value_heads` | 64 | 2 (GQA) | MLA → GQA，group_size=32 |
| | `q_lora_rank` / `kv_lora_rank` | 1536 / 512 | 移除 | GQA 不需要 LoRA 压缩 |
| | `qk_nope_head_dim` | 128 | 移除 | 不变 |
| | `qk_rope_head_dim` | 64 | 移除 | 不变 |
| | `v_head_dim` | 128 | 128 | 不变 |
| | `qk_layernorm` | 未配置 | **True** | **新增**: QK LayerNorm |
| **MoE 架构** | `n_routed_experts` | 384 | 128 | 专家数减少 |
| | `n_shared_experts` | 1 | 1 | 不变 |
| | `num_experts_per_tok` | 8 | 2 | 每次激活 2 个专家 |
| | `first_k_dense_replace` | 1 | 2 | 前 2 层为 Dense FFN |
| | `moe_intermediate_size` | 2048 | **12288** | 单专家维度 ×6 |
| | `intermediate_size` | 18432 | 18432 | Dense FFN 维度不变 |
| | `n_group` / `topk_group` | 1 / 1 | **8 / 2** | **新增**: Grouped Top-K 路由 |
| | `moe_aux_loss_coeff` | 0.001 | **0.01** | 负载均衡系数增大 |
| | `moe_z_loss_coeff` | 未配置 | **0.001** | **新增**: Z-Loss 正则 |
| **位置编码** | `rope_theta` | 50000.0 | 50000.0 | 不变 |
| | `rope_scaling_type` | yarn | yarn | 不变 |
| | `rope_scaling.factor` | - | 32.0 | YaRN 缩放因子 |

#### 3.1.2 模型整体结构

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PCL-1T 模型架构 (32层)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Token Embedding: [163840, 7168] ≈ 1.17B 参数                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Decoder 层 × 32                                │    │
│  │                                                                     │    │
│  │  Layer 0-1: Dense MLP (标准 FFN)                                    │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │    │
│  │  │  GQA Attn    │───►│  Dense FFN   │───►│  RMSNorm     │          │    │
│  │  │  + QK LN     │    │  [18432]     │    │              │          │    │
│  │  └──────────────┘    └──────────────┘    └──────────────┘          │    │
│  │  每层参数量: ~517M                                                  │    │
│  │                              │                                     │    │
│  │                              ▼                                     │    │
│  │  Layer 2-31: MoE (混合专家) × 30 层                                │    │
│  │  ┌──────────────┐    ┌───────────────────────────────────────┐    │    │
│  │  │  GQA Attn    │───►│  MoE Block                            │    │    │
│  │  │  + QK LN     │    │  ┌────────┐ ┌─────────┐ ┌──────────┐ │    │    │
│  │  └──────────────┘    │  │ Router │ │ Shared  │ │ 128×     │ │    │    │
│  │                      │  │[128,   │ │ Expert  │ │ Routed   │ │    │    │
│  │                      │  │ 7168]  │ │[24576,  │ │ Experts  │ │    │    │
│  │                      │  │        │ │ 7168]   │ │[12288]   │ │    │    │
│  │                      │  │        │ │         │ │each      │ │    │    │
│  │                      │  └───┬────┘ └────┬────┘ └────┬─────┘ │    │    │
│  │                      │      └───────────┴───────────┘        │    │    │
│  │                      │              加权合并                   │    │    │
│  │                      └───────────────────────────────────────┘    │    │
│  │  每层参数量: ~34.2B (含全部 128 专家, EP=1)                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Final RMSNorm → LM Head: [163840, 7168] ≈ 1.17B                  │    │
│  │  (LM Head 与 Embedding 不共享权重)                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

参数估算:
├── 词嵌入 + 输出层:     ~2.35B
├── Layer 0-1 (Dense):   ~1.03B (2 × 517M)
├── Layer 2-31 (MoE):    ~1.03T (30 × 34.2B)
└── 总参数量:            ~1.06T

稀疏激活参数 (每次前向, Top-K=2):
├── 激活专家比例: 2/128 = 1.56%
├── MoE 层激活参数:     ~0.5B (30 层 × 每层激活 2 专家)
└── 总激活参数:         ~3.9B (约总参数的 0.4%)
```

---

### 3.2 权重转换 (MCore ↔ HF)

模型训练使用 Megatron-Core (MCore) 分布式格式，推理评测需要先转换为 HuggingFace (HF) 统一格式。

#### 3.2.1 转换原理

训练阶段的分布式并行策略为 **TP=2, PP=8, EP=64** (其中 `moe_tp_extend_ep=True`，实际纯 EP=32)，使用 DualPipeV 调度。转换脚本需要将分散在多个 rank 上的权重碎片收集、拼接为统一的 HF safetensors 格式。

关键挑战：
- **MLA → GQA**: 权重名称和维度映射完全不同，需自定义 `q_proj` / `k_proj` / `v_proj` 投影。
- **MoE 128 专家合并**: 128 个专家的权重分散在不同 EP rank 上，需要正确收集并按序拼接。
- **DualPipeV 调度**: 虚拟流水线阶段会导致权重存储路径与纯 PP 不同，需专门的路径生成逻辑。
- **QK LayerNorm 提取**: 需从 MCore 的 Attention 模块中额外提取 `q_layernorm` / `k_layernorm` 权重。

#### 3.2.2 转换命令

**MCore → HF** (用于推理评测):

```bash
# 默认 DualPipeV 模式（与训练脚本一致）
bash scripts/ckpt_convert_kimi2_mcore2hf.sh

# 指定自定义路径
LOAD_DIR=/path/to/mcore/checkpoint \
SAVE_DIR=/path/to/hf/output \
bash scripts/ckpt_convert_kimi2_mcore2hf.sh

# 后台运行（大模型转换耗时较长）
nohup bash scripts/ckpt_convert_kimi2_mcore2hf.sh \
  > ckpt_convert_kimi2_mcore2hf.log 2>&1 &
```

**HF → MCore** (用于继续训练等场景):

```bash
bash scripts/ckpt_convert_kimi2_hf2mcore.sh
```

#### 3.2.3 关键参数说明

转换脚本 `scripts/ckpt_convert_kimi2_mcore2hf.sh` 的核心参数及与训练配置的对应关系：

| 参数 | 默认值 | 训练脚本值 | 说明 |
|:--|:--|:--|:--|
| `TP` | 2 | 2 | Tensor Parallel |
| `PP` | 8 | 8 | Pipeline Parallel |
| `EP` | 32 | 64 | 纯 EP (训练 EP=64 含 TP, 64/2=32) |
| `NUM_LAYERS` | 32 | 32 | 总层数 |
| `FIRST_K_DENSE_REPLACE` | 2 | 2 | 前 N 层使用 Dense FFN |
| `NUM_EXPERTS` | 128 | 128 | 路由专家数 |
| `QK_LAYERNORM` | 1 | 1 | 启用 QK LayerNorm |
| `MOE_TP_EXTEND_EP` | 1 | 启用 | TP 维度纳入 EP group |
| `SCHEDULES_METHOD` | dualpipev | dualpipev | DualPipeV 调度 |

> **EP 参数说明**: 训练使用 `--expert-model-parallel-size 64`，但由于 `moe_tp_extend_ep=True`，EP group 实际包含 TP 维度，纯 EP = 64 / TP = 32。转换脚本的 `EP` 参数代表纯 EP 维度。


### 3.3 Hugging Face 模型适配

由于 PCL-1T 采用了非标准的 **GQA + QK LayerNorm + Grouped Top-K MoE** 架构，无法直接使用 HF 的 `DeepseekV3ForCausalLM` 原生实现，需要通过 `auto_map` 注册自定义模型代码。

#### 3.3.1 配置文件 (`config.json`)

项目 `models/` 目录包含完整的 HF 格式模型定义，`config.json` 中通过 `auto_map` 指向自定义实现：

```json
{
  "architectures": ["DeepseekV3ForCausalLM"],
  "model_type": "kimi_k2",
  "auto_map": {
    "AutoConfig": "configuration_deepseek.DeepseekV3Config",
    "AutoModel": "modeling_deepseek.DeepseekV3Model",
    "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM"
  },
  "num_query_groups": 2,
  "qk_layernorm": true,
  "n_routed_experts": 128,
  "n_group": 8,
  "topk_group": 2,
  "routed_scaling_factor": 2.827,
  "scoring_func": "sigmoid",
  "topk_method": "noaux_tc"
}
```

#### 3.3.2 关键实现要点

| 组件 | 实现内容 |
|:--|:--|
| **Attention** | GQA（64 heads / 2 KV groups），在 RoPE 之前对 Q/K 应用 RMSNorm（`qk_layernorm=True`），`head_dim=128` |
| **MoE Router** | Grouped Top-K 路由：8 组，每组选 Top-2，最终激活 2 个专家。路由评分使用 sigmoid + 无辅助损失矫正 (`noaux_tc`) |
| **Shared Expert** | 1 个共享专家，`intermediate_size = 12288`，始终参与计算 |
| **Dense FFN** | 前 2 层使用标准 FFN，`intermediate_size = 18432` |
| **RoPE** | YaRN 位置缩放（`factor=32`, `theta=50000`），支持 131K 上下文 |

#### 3.3.3 逐层推理验证

为在单卡 NPU（显存有限）上验证转换后模型的正确性，提供了逐层推理脚本：

```bash
# 逐层推理（峰值显存 ~6GB，适用于验证权重正确性）
python hf_infer/run_inference.py

# 模型结构与权重分析（检测 NaN/Inf、维度异常）
python hf_infer/analyze_model.py
```

---

### 3.4 vLLM 推理框架适配

虽然可通过 `--trust-remote-code` 加载 HF 原生代码进行推理，但这会失去 vLLM 的 **PagedAttention** 和 **FusedMoE** 核心加速特性，导致 OOM 和极差的性能。因此需要在 vLLM 源码级实现原生支持。

#### 3.4.1 源码级适配清单

| 步骤 | 修改文件 | 说明 |
|:--|:--|:--|
| 模型注册 | `vllm/model_executor/models/registry.py` | 添加 `"kimi_k2": ("kimi_k2", "KimiK2ForCausalLM")` |
| Attention | `vllm/model_executor/models/kimi_k2.py` | 实现 GQA + QK LayerNorm，在 RoPE 前应用 RMSNorm |
| MoE | `vllm/model_executor/models/kimi_k2.py` | 集成 `FusedMoE`，实现 Grouped Top-K 路由 + Shared Expert |
| 权重加载 | `vllm/model_executor/models/kimi_k2.py` | `load_weights` 函数中完成 HF → vLLM 权重映射 |

#### 3.4.2 Attention 算子定制要点

```python
# 核心逻辑: QK LayerNorm 必须在 RoPE 之前应用
q = self.q_layernorm(q_proj(hidden_states))  # QK LayerNorm
k = self.k_layernorm(k_proj(hidden_states))  # QK LayerNorm
q, k = self.rotary_emb(q, k, positions)      # RoPE
output = self.attn(q, k, v, ...)              # PagedAttention
```

**昇腾 (Ascend) NPU 特殊处理**: 需确认 FlashAttention 算子是否支持当前 Head 维度配置。若不支持，需在 Python 层对 V 进行 Zero-Padding 补齐，在输出层截断。

#### 3.4.3 MoE 算子定制要点

```python
# Grouped Top-K 路由 + FusedMoE 加速
topk_weights, topk_ids = self.router(hidden_states)  # 8组, 每组 Top-2
routed_out = self.experts(hidden_states, topk_weights, topk_ids)  # FusedMoE
output = routed_out + self.shared_experts(hidden_states)  # 共享专家始终参与
```

**权重映射**: `gate_proj` 和 `up_proj` 按 `dim=0` 拼接加载到 `FusedMoE` 的 `gate_up_proj`；`mlp.gate.weight` 映射到 `router.weight`。

### 4. 踩坑经验与解决方案

在模型从训练到推理的全链路适配过程中，总结以下核心技术挑战及其解决方案：

#### 1. 模型结构对齐 (Model Architecture Alignment)

> **问题背景**: 仅参考训练配置的 shell 脚本，难以理解 Megatron 内部复杂的 Layer 映射，导致转换后权重无法加载。
>
> **对策**: **架构固化**。在训练代码中增加架构打印与参数保存逻辑，通过比对真实参数名与 Shape 强制对齐，而非仅依赖配置推导。

#### 2. 权重转换逻辑
> **问题背景**: 转换脚本若未显式指定 `first_k_dense_replace`，会导致 MoE 层与 Dense 层权重完全错位。
>
> **对策**: **参数强对齐**。建立转换校验机制，确保转换命令中的关键参数与训练脚本 (`pretrain_kimi2_1t_4k.sh`) 严格物理一致。

#### 3. 并行 Rank 映射

> **问题背景**: 开启 `moe_tp_extend_ep` 后，专家权重的存储路径逻辑发生变化，常规转换脚本无法定位文件。
>
> **对策**: **路径函数重构**。专门针对 PCL 架构优化 `_mp_prefix` 路径生成函数，确保在任意 TP/EP 组合下均能精准定位 rank 文件。

#### 4. vLLM 维度冲突 (昇腾适配)

> **问题背景**: 初期在 vLLM 中实现的 Attention 模块不正确，出现了 `112` 等非标准维度，导致在加载参数时昇腾算子报错。
>
> **对策**: **算子对齐**。严格按照 Megatron 内部结构重新实现 Attention 逻辑，确保 Head 维度和Megatron 内部模块对齐。

#### 5. vLLM 框架劫持
> **问题背景**: vLLM 默认加载逻辑受 `config.json` 影响，会强制加载 MLA 模块而非我们实现的 GQA/MoE。
>
> **对策**: **插件化解耦**。引入 **NPUslim 插件系统**，通过 Hook 机制在不侵入修改 vLLM 核心源码的前提下，实现模型组件的正确路由与部署。

#### 6. 基于 MindSpeed-LLM 框架的推理踩坑经验

MindSpeed-LLM 本身也支持 Eval 功能，推理过程中尝试基于 MindSpeed-LLM 进行推理评测。

##### 1. 训练并行策略直接复用导致通信错误

**现象：** 保持和训练的配置（TP=2，PP=8，EP=64，dulepipevc）进行评测，报通信错误。

分析： 训练阶段使用的某些特性（如 `delepipevc` 等训练优化特性）在评估/推理阶段**并未完全支持**，导致通信组初始化失败 

##### 2. 权重转换后 TP=2 报维度错误

**现象：** 将权重转换（TP=2，PP=8，EP=8）的策略进行评测，报维度错误，不支持 TP=2。

##### 3. 权重转换（TP=1，PP=8，EP=8）后无法获得正确答案

**现象：** 权重转换（TP=1，PP=8，EP=8）的策略进行评测，程序能正常运行，但无法获得正确的推理答案。

