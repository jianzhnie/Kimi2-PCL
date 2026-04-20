## Evaluation

PCL-Model 项目在原始开源的 Kimi-K2-Base (`moonshotai/Kimi-K2-Base`) 架构上进行了大量深度改造。本次修改的核心目的是将原先主打推理、采用特殊注意力机制（MLA）的模型架构，调整为**更适合大规模分布式预训练的 GQA 架构**，同时增强了 MoE 的训练特性支持。


## 模型架构对比

为了匹配 PCL-Model 的特定设计，宏观规模参数也发生了明显改变。以下是详细的参数对比表：

| 参数模块       | 参数名称                  | Kimi-K2-Base             | PCL-Model (1T配置)     |
| :------------- | :------------------------ | :----------------------- | :--------------------- |
| **基础规模**   | `num_hidden_layers`       | 61                       | 32                     |
|                | `hidden_size`             | 7168                     | 7168                   |
|                | `vocab_size`              | 163840                   | 163840                 |
|                | `max_position_embeddings` | 131072                   | 131072                 |
| **注意力机制** | `num_attention_heads`     | 64                       | 64                     |
|                | `num_key_value_heads`     | 64                       | 32 (GQA: group_size=2) |
|                | `q_lora_rank`             | 1536                     | (移除MLA)   |
|                | `kv_lora_rank`            | 512                      | (移除MLA)   |
|                | `qk_nope_head_dim`        | 128                      | 128                    |
|                | `qk_rope_head_dim`        | 64                       | 64                     |
|                | `v_head_dim`              | 128                      | 128                    |
|                | `qk_layernorm`            | *未配置*                 | True                   |
| **MoE 架构**   | `n_routed_experts`        | 384                      | 128                    |
|                | `n_shared_experts`        | 1                        | 1                      |
|                | `num_experts_per_tok`     | 8                        | 2                      |
|                | `first_k_dense_replace`   | 1                        | 2                      |
|                | `moe_intermediate_size`   | 2048                     | 12288                  |
|                | `intermediate_size`       | 18432                    | 18432                  |
|                | `n_group`                 | 1                        | 8                      |
|                | `topk_group`              | 1                        | 2                      |
|                | `moe_router_topk`         | *未配置*                 | 2                      |
|                | `moe_aux_loss_coeff`      | 0.001 (`aux_loss_alpha`) | 0.01                   |
|                | `moe_z_loss_coeff`        | *未配置*                 | 0.001                  |
| **位置编码**   | `rope_theta`              | 50000.0                  | 50000.0                |
|                | `rope_scaling_type`       | yarn                     | yarn                   |

*说明：PCL-Model 虽然层数减少，但大幅提高了 MoE 单个专家的维度（从 2048 提升到 12288），从而在较少的路由专家数（128）下支撑起 1T 的参数规模。*



### Kimi2-1T 整体结构

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Kimi2-1T 模型架构 (32层)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         输入处理层                                   │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │  Token Embedding (词嵌入)                                    │   │    │
│  │  │  weight: [163840, 7168] ≈ 1.17B 参数                        │   │    │
│  │  │  vocab_size=163840, hidden_size=7168                        │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Decoder 层 × 32 层                              │    │
│  │                                                                     │    │
│  │  ┌───────────────────────────────────────────────────────────────┐ │    │
│  │  │  Layer 0-1: Dense MLP (标准FFN)                                │ │    │
│  │  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │ │    │
│  │  │  │ Attention   │───►│ Dense FFN   │───►│ LayerNorm   │       │ │    │
│  │  │  │  [8704,7168]│    │ [36864,7168]│    │             │       │ │    │
│  │  │  │  [7168,8192]│    │ [7168,18432]│    │   [7168]    │       │ │    │
│  │  │  └─────────────┘    └─────────────┘    └─────────────┘       │ │    │
│  │  │  每层参数量: ~517M                                              │ │    │
│  │  └───────────────────────────────────────────────────────────────┘ │    │
│  │                              │                                     │    │
│  │                              ▼                                     │    │
│  │  ┌───────────────────────────────────────────────────────────────┐ │    │
│  │  │  Layer 2-31: MoE (混合专家)  × 30层                            │ │    │
│  │  │  ┌─────────────┐    ┌─────────────────────────────────────┐   │ │    │
│  │  │  │ Attention   │───►│  MoE Block                          │   │ │    │
│  │  │  │  (同上)     │    │  ┌─────────┐ ┌─────────┐ ┌────────┐ │   │ │    │
│  │  │  └─────────────┘    │  │ Router  │ │ Shared  │ │ Routed │ │   │ │    │
│  │  │                     │  │ [128,   │ │ Expert  │ │Expert  │ │   │ │    │
│  │  │                     │  │  7168]  │ │[24576,  │ │[7168,  │ │   │ │    │
│  │  │                     │  │         │ │ 7168]   │ │3145728]│ │   │ │    │
│  │  │                     │  │         │ │(fc1)    │ │(W1,128 │ │   │ │    │
│  │  │                     │  │         │ │         │ │experts)│ │   │ │    │
│  │  │                     │  └────┬────┘ └────┬────┘ └───┬────┘ │   │ │    │
│  │  │                     │       │           │          │      │   │ │    │
│  │  │                     │       └───────────┴──────────┘      │   │ │    │
│  │  │                     │                   │                 │   │ │    │
│  │  │                     │              加权合并                 │   │ │    │
│  │  │                     └─────────────────────────────────────┘   │ │    │
│  │  │  每层参数量: ~34.2B (含全部128专家, EP=1 dump)                 │ │    │
│  │  └───────────────────────────────────────────────────────────────┘ │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        输出处理层                                    │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │  Final LayerNorm                                             │   │    │
│  │  │  weight: [7168]                                             │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  │                              │                                      │    │
│  │                              ▼                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │  LM Head (输出层)                                            │   │    │
│  │  │  weight: [163840, 7168] ≈ 1.17B 参数                        │   │    │
│  │  │  (与Embedding不共享)                                        │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

完整 32 层模型估算 (EP=1):
├── 词嵌入 + 输出层:    ~2.35B
├── Layer 0-1 (Dense):  ~1.03B (2 × 517M)
├── Layer 2-31 (MoE):   ~1.03T (30 × 34.2B)
└── 总计:               ~1.06T 参数

稀疏激活参数 (每次前向, EP=64):
├── 激活专家数: 2/128 = 1.56%
├── MoE 层激活参数:     ~0.5B (30 × 16.8M, 每token激活2专家)
└── 总计激活参数:       ~3.9B (约 4% 的总参数, 推理时)
```


### 权重转换


#### Step1： Mcore ===> Huggingface 

支持多种并行维度和特性的转换，包括：

- **TP (Tensor Parallel)**: 张量并行规模。
- **PP (Pipeline Parallel)**: 流水线并行规模。
- **EP (Expert Parallel)**: 专家并行规模。
- **VPP (Virtual Pipeline Parallel)**: 虚拟流水线并行，通过 `vpp-stage` 配置每个虚拟阶段的层数，减少流水线气泡。
- **dualpipev**: 开启 DualPipe 调度支持。启用后，`vpp_size` 默认为 2，且要求每个流水线阶段的层数为偶数。
- **MOE-TP-EXTEND-EP**: MoE 并行扩展策略。启用后，TP 组将参与专家并行，允许在保持 TP 的同时增加专家并行度（与 `expert-tp-size > 1` 互斥）。

```bash
mp_rank_00_000_000  mp_rank_00_003_014  mp_rank_00_006_028  mp_rank_01_001_043  mp_rank_01_004_057
mp_rank_00_000_002  mp_rank_00_003_016  mp_rank_00_006_030  mp_rank_01_001_045  mp_rank_01_004_059
mp_rank_00_000_004  mp_rank_00_003_018  mp_rank_00_006_032  mp_rank_01_001_047  mp_rank_01_004_061
mp_rank_00_000_006  mp_rank_00_003_020  mp_rank_00_006_034  mp_rank_01_001_049  mp_rank_01_004_063
mp_rank_00_000_008  mp_rank_00_003_022  mp_rank_00_006_036  mp_rank_01_001_051  mp_rank_01_005_001
mp_rank_00_000_010  mp_rank_00_003_024  mp_rank_00_006_038  mp_rank_01_001_053  mp_rank_01_005_003
mp_rank_00_000_012  mp_rank_00_003_026  mp_rank_00_006_040  mp_rank_01_001_055  mp_rank_01_005_005
mp_rank_00_000_014  mp_rank_00_003_028  mp_rank_00_006_042  mp_rank_01_001_057  mp_rank_01_005_007
mp_rank_00_000_016  mp_rank_00_003_030  mp_rank_00_006_044  mp_rank_01_001_059  mp_rank_01_005_009
mp_rank_00_000_018  mp_rank_00_003_032  mp_rank_00_006_046  mp_rank_01_001_061  mp_rank_01_005_011
mp_rank_00_000_020  mp_rank_00_003_034  mp_rank_00_006_048  mp_rank_01_001_063  mp_rank_01_005_013
mp_rank_00_000_022  mp_rank_00_003_036  mp_rank_00_006_050  mp_rank_01_002_001  mp_rank_01_005_015
mp_rank_00_000_024  mp_rank_00_003_038  mp_rank_00_006_052  mp_rank_01_002_003  mp_rank_01_005_017
mp_rank_00_000_026  mp_rank_00_003_040  mp_rank_00_006_054  mp_rank_01_002_005  mp_rank_01_005_019
mp_rank_00_000_028  mp_rank_00_003_042  mp_rank_00_006_056  mp_rank_01_002_007  mp_rank_01_005_021
mp_rank_00_000_030  mp_rank_00_003_044  mp_rank_00_006_058  mp_rank_01_002_009  mp_rank_01_005_023
mp_rank_00_000_032  mp_rank_00_003_046  mp_rank_00_006_060  mp_rank_01_002_011  mp_rank_01_005_025
mp_rank_00_000_034  mp_rank_00_003_048  mp_rank_00_006_062  mp_rank_01_002_013  mp_rank_01_005_027
mp_rank_00_000_036  mp_rank_00_003_050  mp_rank_00_007_000  mp_rank_01_002_015  mp_rank_01_005_029
mp_rank_00_000_038  mp_rank_00_003_052  mp_rank_00_007_002  mp_rank_01_002_017  mp_rank_01_005_031
mp_rank_00_000_040  mp_rank_00_003_054  mp_rank_00_007_004  mp_rank_01_002_019  mp_rank_01_005_033
mp_rank_00_000_042  mp_rank_00_003_056  mp_rank_00_007_006  mp_rank_01_002_021  mp_rank_01_005_035
mp_rank_00_000_044  mp_rank_00_003_058  mp_rank_00_007_008  mp_rank_01_002_023  mp_rank_01_005_037
mp_rank_00_000_046  mp_rank_00_003_060  mp_rank_00_007_010  mp_rank_01_002_025  mp_rank_01_005_039
mp_rank_00_000_048  mp_rank_00_003_062  mp_rank_00_007_012  mp_rank_01_002_027  mp_rank_01_005_041
mp_rank_00_000_050  mp_rank_00_004_000  mp_rank_00_007_014  mp_rank_01_002_029  mp_rank_01_005_043
mp_rank_00_000_052  mp_rank_00_004_002  mp_rank_00_007_016  mp_rank_01_002_031  mp_rank_01_005_045
mp_rank_00_000_054  mp_rank_00_004_004  mp_rank_00_007_018  mp_rank_01_002_033  mp_rank_01_005_047
mp_rank_00_000_056  mp_rank_00_004_006  mp_rank_00_007_020  mp_rank_01_002_035  mp_rank_01_005_049
mp_rank_00_000_058  mp_rank_00_004_008  mp_rank_00_007_022  mp_rank_01_002_037  mp_rank_01_005_051
mp_rank_00_000_060  mp_rank_00_004_010  mp_rank_00_007_024  mp_rank_01_002_039  mp_rank_01_005_053
mp_rank_00_000_062  mp_rank_00_004_012  mp_rank_00_007_026  mp_rank_01_002_041  mp_rank_01_005_055
mp_rank_00_001_000  mp_rank_00_004_014  mp_rank_00_007_028  mp_rank_01_002_043  mp_rank_01_005_057
mp_rank_00_001_002  mp_rank_00_004_016  mp_rank_00_007_030  mp_rank_01_002_045  mp_rank_01_005_059
mp_rank_00_001_004  mp_rank_00_004_018  mp_rank_00_007_032  mp_rank_01_002_047  mp_rank_01_005_061
mp_rank_00_001_006  mp_rank_00_004_020  mp_rank_00_007_034  mp_rank_01_002_049  mp_rank_01_005_063
mp_rank_00_001_008  mp_rank_00_004_022  mp_rank_00_007_036  mp_rank_01_002_051  mp_rank_01_006_001
mp_rank_00_001_010  mp_rank_00_004_024  mp_rank_00_007_038  mp_rank_01_002_053  mp_rank_01_006_003
mp_rank_00_001_012  mp_rank_00_004_026  mp_rank_00_007_040  mp_rank_01_002_055  mp_rank_01_006_005
mp_rank_00_001_014  mp_rank_00_004_028  mp_rank_00_007_042  mp_rank_01_002_057  mp_rank_01_006_007
mp_rank_00_001_016  mp_rank_00_004_030  mp_rank_00_007_044  mp_rank_01_002_059  mp_rank_01_006_009
mp_rank_00_001_018  mp_rank_00_004_032  mp_rank_00_007_046  mp_rank_01_002_061  mp_rank_01_006_011
mp_rank_00_001_020  mp_rank_00_004_034  mp_rank_00_007_048  mp_rank_01_002_063  mp_rank_01_006_013
mp_rank_00_001_022  mp_rank_00_004_036  mp_rank_00_007_050  mp_rank_01_003_001  mp_rank_01_006_015
mp_rank_00_001_024  mp_rank_00_004_038  mp_rank_00_007_052  mp_rank_01_003_003  mp_rank_01_006_017
mp_rank_00_001_026  mp_rank_00_004_040  mp_rank_00_007_054  mp_rank_01_003_005  mp_rank_01_006_019
mp_rank_00_001_028  mp_rank_00_004_042  mp_rank_00_007_056  mp_rank_01_003_007  mp_rank_01_006_021
mp_rank_00_001_030  mp_rank_00_004_044  mp_rank_00_007_058  mp_rank_01_003_009  mp_rank_01_006_023
mp_rank_00_001_032  mp_rank_00_004_046  mp_rank_00_007_060  mp_rank_01_003_011  mp_rank_01_006_025
mp_rank_00_001_034  mp_rank_00_004_048  mp_rank_00_007_062  mp_rank_01_003_013  mp_rank_01_006_027
mp_rank_00_001_036  mp_rank_00_004_050  mp_rank_01_000_001  mp_rank_01_003_015  mp_rank_01_006_029
mp_rank_00_001_038  mp_rank_00_004_052  mp_rank_01_000_003  mp_rank_01_003_017  mp_rank_01_006_031
mp_rank_00_001_040  mp_rank_00_004_054  mp_rank_01_000_005  mp_rank_01_003_019  mp_rank_01_006_033
mp_rank_00_001_042  mp_rank_00_004_056  mp_rank_01_000_007  mp_rank_01_003_021  mp_rank_01_006_035
mp_rank_00_001_044  mp_rank_00_004_058  mp_rank_01_000_009  mp_rank_01_003_023  mp_rank_01_006_037
mp_rank_00_001_046  mp_rank_00_004_060  mp_rank_01_000_011  mp_rank_01_003_025  mp_rank_01_006_039
mp_rank_00_001_048  mp_rank_00_004_062  mp_rank_01_000_013  mp_rank_01_003_027  mp_rank_01_006_041
mp_rank_00_001_050  mp_rank_00_005_000  mp_rank_01_000_015  mp_rank_01_003_029  mp_rank_01_006_043
mp_rank_00_001_052  mp_rank_00_005_002  mp_rank_01_000_017  mp_rank_01_003_031  mp_rank_01_006_045
mp_rank_00_001_054  mp_rank_00_005_004  mp_rank_01_000_019  mp_rank_01_003_033  mp_rank_01_006_047
mp_rank_00_001_056  mp_rank_00_005_006  mp_rank_01_000_021  mp_rank_01_003_035  mp_rank_01_006_049
mp_rank_00_001_058  mp_rank_00_005_008  mp_rank_01_000_023  mp_rank_01_003_037  mp_rank_01_006_051
mp_rank_00_001_060  mp_rank_00_005_010  mp_rank_01_000_025  mp_rank_01_003_039  mp_rank_01_006_053
mp_rank_00_001_062  mp_rank_00_005_012  mp_rank_01_000_027  mp_rank_01_003_041  mp_rank_01_006_055
mp_rank_00_002_000  mp_rank_00_005_014  mp_rank_01_000_029  mp_rank_01_003_043  mp_rank_01_006_057
mp_rank_00_002_002  mp_rank_00_005_016  mp_rank_01_000_031  mp_rank_01_003_045  mp_rank_01_006_059
mp_rank_00_002_004  mp_rank_00_005_018  mp_rank_01_000_033  mp_rank_01_003_047  mp_rank_01_006_061
mp_rank_00_002_006  mp_rank_00_005_020  mp_rank_01_000_035  mp_rank_01_003_049  mp_rank_01_006_063
mp_rank_00_002_008  mp_rank_00_005_022  mp_rank_01_000_037  mp_rank_01_003_051  mp_rank_01_007_001
mp_rank_00_002_010  mp_rank_00_005_024  mp_rank_01_000_039  mp_rank_01_003_053  mp_rank_01_007_003
mp_rank_00_002_012  mp_rank_00_005_026  mp_rank_01_000_041  mp_rank_01_003_055  mp_rank_01_007_005
mp_rank_00_002_014  mp_rank_00_005_028  mp_rank_01_000_043  mp_rank_01_003_057  mp_rank_01_007_007
mp_rank_00_002_016  mp_rank_00_005_030  mp_rank_01_000_045  mp_rank_01_003_059  mp_rank_01_007_009
mp_rank_00_002_018  mp_rank_00_005_032  mp_rank_01_000_047  mp_rank_01_003_061  mp_rank_01_007_011
mp_rank_00_002_020  mp_rank_00_005_034  mp_rank_01_000_049  mp_rank_01_003_063  mp_rank_01_007_013
mp_rank_00_002_022  mp_rank_00_005_036  mp_rank_01_000_051  mp_rank_01_004_001  mp_rank_01_007_015
mp_rank_00_002_024  mp_rank_00_005_038  mp_rank_01_000_053  mp_rank_01_004_003  mp_rank_01_007_017
mp_rank_00_002_026  mp_rank_00_005_040  mp_rank_01_000_055  mp_rank_01_004_005  mp_rank_01_007_019
mp_rank_00_002_028  mp_rank_00_005_042  mp_rank_01_000_057  mp_rank_01_004_007  mp_rank_01_007_021
mp_rank_00_002_030  mp_rank_00_005_044  mp_rank_01_000_059  mp_rank_01_004_009  mp_rank_01_007_023
mp_rank_00_002_032  mp_rank_00_005_046  mp_rank_01_000_061  mp_rank_01_004_011  mp_rank_01_007_025
mp_rank_00_002_034  mp_rank_00_005_048  mp_rank_01_000_063  mp_rank_01_004_013  mp_rank_01_007_027
mp_rank_00_002_036  mp_rank_00_005_050  mp_rank_01_001_001  mp_rank_01_004_015  mp_rank_01_007_029
mp_rank_00_002_038  mp_rank_00_005_052  mp_rank_01_001_003  mp_rank_01_004_017  mp_rank_01_007_031
mp_rank_00_002_040  mp_rank_00_005_054  mp_rank_01_001_005  mp_rank_01_004_019  mp_rank_01_007_033
mp_rank_00_002_042  mp_rank_00_005_056  mp_rank_01_001_007  mp_rank_01_004_021  mp_rank_01_007_035
mp_rank_00_002_044  mp_rank_00_005_058  mp_rank_01_001_009  mp_rank_01_004_023  mp_rank_01_007_037
mp_rank_00_002_046  mp_rank_00_005_060  mp_rank_01_001_011  mp_rank_01_004_025  mp_rank_01_007_039
mp_rank_00_002_048  mp_rank_00_005_062  mp_rank_01_001_013  mp_rank_01_004_027  mp_rank_01_007_041
mp_rank_00_002_050  mp_rank_00_006_000  mp_rank_01_001_015  mp_rank_01_004_029  mp_rank_01_007_043
mp_rank_00_002_052  mp_rank_00_006_002  mp_rank_01_001_017  mp_rank_01_004_031  mp_rank_01_007_045
mp_rank_00_002_054  mp_rank_00_006_004  mp_rank_01_001_019  mp_rank_01_004_033  mp_rank_01_007_047
mp_rank_00_002_056  mp_rank_00_006_006  mp_rank_01_001_021  mp_rank_01_004_035  mp_rank_01_007_049
mp_rank_00_002_058  mp_rank_00_006_008  mp_rank_01_001_023  mp_rank_01_004_037  mp_rank_01_007_051
mp_rank_00_002_060  mp_rank_00_006_010  mp_rank_01_001_025  mp_rank_01_004_039  mp_rank_01_007_053
mp_rank_00_002_062  mp_rank_00_006_012  mp_rank_01_001_027  mp_rank_01_004_041  mp_rank_01_007_055
mp_rank_00_003_000  mp_rank_00_006_014  mp_rank_01_001_029  mp_rank_01_004_043  mp_rank_01_007_057
mp_rank_00_003_002  mp_rank_00_006_016  mp_rank_01_001_031  mp_rank_01_004_045  mp_rank_01_007_059
mp_rank_00_003_004  mp_rank_00_006_018  mp_rank_01_001_033  mp_rank_01_004_047  mp_rank_01_007_061
mp_rank_00_003_006  mp_rank_00_006_020  mp_rank_01_001_035  mp_rank_01_004_049  mp_rank_01_007_063
mp_rank_00_003_008  mp_rank_00_006_022  mp_rank_01_001_037  mp_rank_01_004_051
mp_rank_00_003_010  mp_rank_00_006_024  mp_rank_01_001_039  mp_rank_01_004_053
mp_rank_00_003_012  mp_rank_00_006_026  mp_rank_01_001_041  mp_rank_01_004_055
```


#### Step2： Huggingface ===> Mcore

将 Huggingface 格式的权重转换为分布式 MCore 检查点，支持以下关键配置：

- **并行策略映射**: 转换时需指定目标集群的 `--target-tensor-parallel-size` (TP), `--target-pipeline-parallel-size` (PP) 和 `--target-expert-parallel-size` (EP)。
- **混合架构处理**: 必须通过 `--first-k-dense-replace 2` 明确指定前 2 层为密集层，以确保权重切分逻辑的正确性。
- **调度算法对齐**: 若目标训练环境使用 DualPipe，需指定 `--schedules-method dualpipev`。
- **高效转换**: 支持 `--hf-io-threads` 多线程读取和 `--pp-workers` 多进程并行处理不同流水线阶段，显著提升转换速度。

**示例转换命令**:
```bash
python utils/convert_kimi2_hf2mcore.py \
    --load-dir ./path/to/hf_model \
    --save-dir ./path/to/mcore_ckpt \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 8 \
    --target-expert-parallel-size 64 \
    --num-layers 32 \
    --first-k-dense-replace 2 \
    --schedules-method dualpipev \
    --hf-io-threads 16
```


### 难点

1. 权重转换时，对齐模型架构和 并行策略的映射关系，确保在目标集群上能够正确运行。
2. 
