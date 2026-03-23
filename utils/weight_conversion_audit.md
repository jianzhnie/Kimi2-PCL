# 权重转换脚本系统性审查与优化报告

## 一、模型架构参数对比分析

### 1.1 Pretrain脚本核心参数 (pretrain_kimi2_1t_4k.sh)

| 参数类别       | 参数名称                            | 值        | 说明               |
| -------------- | ----------------------------------- | --------- | ------------------ |
| **基础架构**   | NUM_LAYERS                          | 32        | 总层数             |
|                | HIDDEN_SIZE                         | 7168      | 隐藏维度           |
|                | FFN_HIDDEN_SIZE                     | 18432     | Dense FFN中间维度  |
|                | MOE_FFN_HIDDEN_SIZE                 | 12288     | MoE Expert中间维度 |
|                | VOCAB_SIZE                          | 163840    | 词表大小           |
|                | MAX_POSITION_EMBEDDINGS             | 131072    | 最大位置编码       |
| **并行策略**   | TP                                  | 2         | 张量并行           |
|                | PP                                  | 8         | 流水线并行         |
|                | EP                                  | 64        | 专家并行           |
|                | CP                                  | 1         | 上下文并行         |
|                | SCHEDULES_METHOD                    | dualpipev | 双向流水线调度     |
| **注意力机制** | NUM_ATTENTION_HEADS                 | 64        | 注意力头数         |
|                | NUM_QUERY_GROUPS                    | 2         | GQA组数            |
|                | KV_CHANNELS                         | 128       | KV通道维度         |
|                | QK_LAYERNORM                        | True      | 启用QK LayerNorm   |
|                | USE_FLASH_ATTN                      | True      | 使用FlashAttention |
| **头维度**     | QK_HEAD_DIM (推断)                  | 128       | QK nope维度        |
|                | QK_POS_EMB_HEAD_DIM (推断)          | 64        | QK rope维度        |
|                | V_HEAD_DIM (推断)                   | 128       | V头维度            |
| **RoPE配置**   | ROTARY_BASE                         | 50000     | RoPE base          |
|                | ROPE_SCALING_TYPE                   | yarn      | RoPE缩放类型       |
|                | ROPE_SCALING_FACTOR                 | 32        | 缩放因子           |
|                | ORIGINAL_MAX_POSITION_EMBEDDINGS    | 4096      | 原始最大位置编码   |
| **MoE配置**    | NUM_EXPERTS                         | 128       | 专家数量           |
|                | FIRST_K_DENSE_REPLACE               | 2         | 前K层为Dense       |
|                | N_SHARED_EXPERTS                    | 1         | 共享专家数         |
|                | MOE_ROUTER_TOPK                     | 2         | Router top-k       |
|                | MOE_GROUPED_GEMM                    | True      | 使用grouped gemm   |
|                | MOE_ROUTER_DTYPE                    | fp32      | Router计算精度     |
|                | MOE_ROUTER_ENABLE_EXPERT_BIAS       | True      | 启用expert bias    |
| **激活函数**   | SWIGLU                              | True      | 使用SwiGLU         |
|                | USE_FUSED_SWIGLU                    | True      | 使用融合SwiGLU     |
| **其他**       | UNTIE_EMBEDDINGS_AND_OUTPUT_WEIGHTS | True      | 解耦embed和output  |
|                | NORMALIZATION                       | RMSNorm   | 归一化类型         |
|                | NORM_EPSILON                        | 1e-6      | 归一化epsilon      |
|                | BF16                                | True      | BF16精度           |
|                | SEQUENCE_PARALLEL                   | True      | 序列并行           |
|                | USE_ROTARY_POSITION_EMBEDDINGS      | True      | 使用旋转位置编码   |

### 1.2 转换脚本默认参数对比

| 参数                    | convert_ckpt_hf2mcore.py | convert_ckpt_mcore2hf.py | ckpt_convert_*.sh | 是否一致 |
| ----------------------- | ------------------------ | ------------------------ | ----------------- | -------- |
| HIDDEN_SIZE             | 7168 ✓                   | 7168 ✓                   | 7168 ✓            | ✅        |
| NUM_EXPERTS             | 128 ✓                    | 128 ✓                    | 128 ✓             | ✅        |
| FIRST_K_DENSE_REPLACE   | 2 ✓                      | 2 ✓                      | 2 ✓               | ✅        |
| NUM_ATTENTION_HEADS     | 64 ✓                     | 64 ✓                     | 64 ✓              | ✅        |
| QK_HEAD_DIM             | 128 ✓                    | 128 ✓                    | 128 ✓             | ✅        |
| V_HEAD_DIM              | 128 ✓                    | 128 ✓                    | 128 ✓             | ✅        |
| QK_POS_EMB_HEAD_DIM     | 64 ✓                     | 64 ✓                     | 64 ✓              | ✅        |
| VOCAB_SIZE              | 163840 ✓                 | 163840 ✓                 | 163840 ✓          | ✅        |
| ROTARY_BASE             | 50000 ✓                  | 50000 ✓                  | 50000 ✓           | ✅        |
| MOE_FFN_HIDDEN_SIZE     | 可从 HF config 推断      | 可选（用于写出 config）  | 12288 ✓           | ✅        |
| FFN_HIDDEN_SIZE         | 可从 HF config 推断      | 可选（用于写出 config）  | 18432 ✓           | ✅        |
| MAX_POSITION_EMBEDDINGS | 当前未使用               | 可选（用于写出 config）  | 131072 ✓          | ✅        |
| NUM_QUERY_GROUPS        | 当前未使用               | 可选（用于写出 config）  | 2 ✓               | ✅        |
