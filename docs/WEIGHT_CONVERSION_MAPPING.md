# Kimi2-PCL 权重转换映射与验证文档

本文档系统性地梳理了 Kimi2-PCL 项目中 Kimi2-1T 模型在 Megatron-Core 与 Hugging Face 格式之间的权重转换规则、参数映射关系以及架构一致性验证细节。

## 1. 模型架构与参数配置

### 1.1 预训练核心参数 (Kimi2-1T)
基于 `pretrain_kimi2_1t_4k.sh` 脚本提取的基准模型架构参数：

| 参数模块       | 参数名称                     | 设定值     | 架构说明                          |
| :------------- | :--------------------------- | :--------- | :-------------------------------- |
| **基础规模**   | `NUM_LAYERS`                 | 32         | Transformer 总层数                |
|                | `HIDDEN_SIZE`                | 7168       | 隐藏层维度                        |
|                | `VOCAB_SIZE`                 | 163840     | 词表大小                          |
|                | `MAX_POSITION_EMBEDDINGS`    | 131072     | 最大支持上下文长度                |
| **注意力机制** | `NUM_ATTENTION_HEADS`        | 64         | Query 头数                        |
|                | `NUM_QUERY_GROUPS`           | 2          | GQA 分组比（推断 KV 头数为 32）   |
|                | `QK_HEAD_DIM` / `V_HEAD_DIM` | 128 / 128  | QK nope 维度 / V 维度             |
|                | `QK_POS_EMB_HEAD_DIM`        | 64         | QK rope 维度                      |
|                | `QK_LAYERNORM`               | True       | 启用 QK 独立 LayerNorm            |
| **MoE 架构**   | `NUM_EXPERTS`                | 128        | 路由专家总数                      |
|                | `N_SHARED_EXPERTS`           | 1          | 共享专家数量                      |
|                | `FIRST_K_DENSE_REPLACE`      | 2          | 前 2 层使用 Dense MLP，不使用 MoE |
|                | `MOE_FFN_HIDDEN_SIZE`        | 12288      | 专家网络的中间层维度              |
|                | `FFN_HIDDEN_SIZE`            | 18432      | Dense 层的中间层维度              |
|                | `MOE_ROUTER_TOPK`            | 2          | 路由 Top-K 激活数                 |
| **位置编码**   | `ROTARY_BASE`                | 50000      | RoPE 基础频率                     |
|                | `ROPE_SCALING_TYPE`          | yarn       | RoPE 缩放类型 (Factor: 32)        |
| **并行策略**   | `TP` / `PP` / `EP`           | 2 / 8 / 64 | 张量并行 / 流水线并行 / 专家并行  |
|                | `SCHEDULES_METHOD`           | dualpipev  | 双向流水线调度策略                |

### 1.2 转换脚本参数对齐状态
转换工具 (`convert_ckpt_hf2mcore.py` / `convert_ckpt_mcore2hf.py`) 的默认配置与预训练脚本的一致性：

| 核心参数                | 转换脚本设定          | 预训练脚本设定    | 对齐状态 |
| :---------------------- | :-------------------- | :---------------- | :------: |
| `HIDDEN_SIZE`           | 7168                  | 7168              |    ✅     |
| `NUM_EXPERTS`           | 128                   | 128               |    ✅     |
| `FIRST_K_DENSE_REPLACE` | 2                     | 2                 |    ✅     |
| `NUM_ATTENTION_HEADS`   | 64                    | 64                |    ✅     |
| `QK/V/ROPE_HEAD_DIM`    | 128 / 128 / 64        | 128 / 128 / 64    |    ✅     |
| `VOCAB_SIZE`            | 163840                | 163840            |    ✅     |
| `ROTARY_BASE`           | 50000                 | 50000             |    ✅     |
| `NUM_KEY_VALUE_HEADS`   | 32 (转换脚本显式使用) | 64 / 2 = 32 (GQA) |    ✅     |

---

## 2. 权重参数映射字典

Megatron-Core 与 Hugging Face 的权重命名及张量排布存在显著差异。以下为双向转换的映射规则：

### 2.1 基础网络层映射

| Megatron-Core (MCore)                                  | Hugging Face (HF)                                  | 形状变换与操作       |
| :----------------------------------------------------- | :------------------------------------------------- | :------------------- |
| `embedding.word_embeddings.weight`                     | `model.embed_tokens.weight`                        | TP 合并/分割 (dim=0) |
| `output_layer.weight`                                  | `lm_head.weight`                                   | TP 合并/分割 (dim=0) |
| `decoder.final_layernorm.weight`                       | `model.norm.weight`                                | 无变换               |
| `decoder.layers.{i}.input_layernorm.weight`            | `model.layers.{i}.input_layernorm.weight`          | 无变换               |
| `decoder.layers.{i}.pre_mlp_layernorm.weight`          | `model.layers.{i}.post_attention_layernorm.weight` | 无变换               |
| `decoder.layers.{i}.self_attention.q_layernorm.weight` | `model.layers.{i}.self_attn.q_layernorm.weight`    | 无变换               |
| `decoder.layers.{i}.self_attention.k_layernorm.weight` | `model.layers.{i}.self_attn.k_layernorm.weight`    | 无变换               |
| `decoder.layers.{i}.self_attention.linear_proj.weight` | `model.layers.{i}.self_attn.o_proj.weight`         | TP 合并/分割 (dim=1) |

### 2.2 注意力层 (Attention QKV)

| Megatron-Core (MCore)                                 | Hugging Face (HF)                                                                                                                          | 形状变换与操作                                                                                         |
| :---------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| `decoder.layers.{i}.self_attention.linear_qkv.weight` | `model.layers.{i}.self_attn.q_proj.weight` <br> `model.layers.{i}.self_attn.k_proj.weight` <br> `model.layers.{i}.self_attn.v_proj.weight` | **MCore→HF**: 沿 dim=0 切分为 Q/K/V，再按 TP 合并 <br> **HF→MCore**: 沿 dim=0 按 TP 切分，再拼接 Q+K+V |

### 2.3 Dense MLP 层 (前 `first_k_dense_replace` 层)

| Megatron-Core (MCore)                      | Hugging Face (HF)                                                                  | 形状变换与操作         |
| :----------------------------------------- | :--------------------------------------------------------------------------------- | :--------------------- |
| `decoder.layers.{i}.mlp.linear_fc1.weight` | `model.layers.{i}.mlp.gate_proj.weight` <br> `model.layers.{i}.mlp.up_proj.weight` | 沿 dim=0 对半切分/拼接 |
| `decoder.layers.{i}.mlp.linear_fc2.weight` | `model.layers.{i}.mlp.down_proj.weight`                                            | TP 合并/分割 (dim=1)   |

### 2.4 MoE 专家层 (后续层)

| Megatron-Core (MCore)                                     | Hugging Face (HF)                                                                                              | 形状变换与操作         |
| :-------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- | :--------------------- |
| `decoder.layers.{i}.mlp.router.weight`                    | `model.layers.{i}.mlp.gate.weight`                                                                             | 跨 EP 合并/分割        |
| `decoder.layers.{i}.mlp.router.expert_bias`               | `model.layers.{i}.mlp.gate.e_score_correction_bias`                                                            | 跨 EP 合并/分割        |
| `decoder.layers.{i}.mlp.shared_experts.linear_fc1.weight` | `model.layers.{i}.mlp.shared_experts.gate_proj.weight`<br>`model.layers.{i}.mlp.shared_experts.up_proj.weight` | 切分 gate/up + TP 变换 |
| `decoder.layers.{i}.mlp.shared_experts.linear_fc2.weight` | `model.layers.{i}.mlp.shared_experts.down_proj.weight`                                                         | TP 变换                |
| `decoder.layers.{i}.mlp.experts.weight1` (Grouped)        | `model.layers.{i}.mlp.experts.{e}.gate_proj.weight`<br>`model.layers.{i}.mlp.experts.{e}.up_proj.weight`       | 见 3.2 专家重组逻辑    |
| `decoder.layers.{i}.mlp.experts.weight2` (Grouped)        | `model.layers.{i}.mlp.experts.{e}.down_proj.weight`                                                            | 见 3.2 专家重组逻辑    |

---

## 3. 核心张量变换逻辑详解

### 3.1 QKV 分离与合并逻辑

Megatron-Core 采用连续内存存储 QKV，而 Hugging Face 将其独立为三个投影矩阵。在 Kimi2-1T (GQA) 架构下：
*   **Q_head_dim** = `qk_nope` (128) + `qk_rope` (64) = 192
*   **K_head_dim** = `qk_nope` (128) + `qk_rope` (64) = 192
*   **V_head_dim** = 128

**MCore → HF (提取与合并)**
```python
# 1. 计算每个 TP Rank 内的头部数量
heads_per_tp = num_attention_heads // tp_size           # 64 / 2 = 32
kv_heads_per_tp = num_key_value_heads // tp_size        # 32 / 2 = 16

# 2. 计算每个 TP Rank 内张量的行数
q_per_tp = heads_per_tp * q_head_dim                    # 32 * 192
k_per_tp = kv_heads_per_tp * q_head_dim                 # 16 * 192
v_per_tp = kv_heads_per_tp * v_head_dim                 # 16 * 128

# 3. 切分单个 TP 的 QKV
q_r, k_r, v_r = torch.split(qkv_shard, [q_per_tp, k_per_tp, v_per_tp], dim=0)

# 4. 合并所有 TP 分片
q_proj = torch.cat(q_parts, dim=0)  # 最终 [64 * 192, 7168]
k_proj = torch.cat(k_parts, dim=0)  # 最终 [32 * 192, 7168]
v_proj = torch.cat(v_parts, dim=0)  # 最终 [32 * 128, 7168]
```

### 3.2 Grouped GEMM 专家权重重组

当开启 `--moe-grouped-gemm` 时，Megatron 将当前 EP Rank 负责的所有局部专家（`num_local_experts`）权重在连续内存中合并存储。

**MCore 存储格式:**
*   `experts.weight1`: 形状为 `[hidden_size, num_local * (intermediate * 2)]`
*   `experts.weight2`: 形状为 `[num_local * intermediate, hidden_size]`

**还原至 HF 的独立专家 (MCore → HF):**
```python
num_local = num_experts // ep_size   # 128 / 64 = 2

# 处理 weight1 (gate + up)
w1_3d = local_w1.view(hidden_size, num_local, -1).permute(1, 0, 2)  
for li in range(num_local):
    fc1 = w1_3d[li].t()               # 恢复为 [intermediate*2, hidden_size]
    gate, up = torch.chunk(fc1, 2, dim=0)  # 切分为 gate_proj 和 up_proj

# 处理 weight2 (down)
w2_3d = local_w2.view(num_local, -1, hidden_size)  
for li in range(num_local):
    down = w2_3d[li].t()              # 恢复为 [hidden_size, intermediate]
    # 存为 down_proj
```

---

## 4. 1T 参数量规模统计验证

### 4.1 理论参数量分析 (全量模型)
基于 Kimi2-1T 架构的估算（不含极小量如 bias）：

| 模型组件              | 计算公式                                              | 参数量 (约为) |
| :-------------------- | :---------------------------------------------------- | :------------ |
| **词嵌入/输出**       | `163840 * 7168 * 2` (未绑定)                          | 2.34 B        |
| **Attention 层**      | `32 * [ (64*192 + 32*192 + 32*128 + 64*128) * 7168 ]` | ~7.01 B       |
| **Dense MLP (前2层)** | `2 * [ 3 * 7168 * 18432 ]`                            | ~0.79 B       |
| **MoE Shared (30层)** | `30 * [ 3 * 7168 * 12288 ]`                           | ~7.92 B       |
| **MoE Routed (30层)** | `30 * 128 * [ 3 * 7168 * 12288 ]`                     | ~1,014.6 B    |
| **总计估算**          |                                                       | **~1.03 T**   |

### 4.2 检查点分片体积 (TP=2, PP=8, EP=64)
在分布式训练/转换中，单张卡的显存占用由以下切分决定：

*   **PP 切分**: 每卡负责 `32 / 8 = 4` 层。
*   **TP 切分**: 词嵌入、输出层、Attention QKV/O、MLP 参数均被分为 2 份。
*   **EP 切分**: 每卡仅负责 `128 / 64 = 2` 个路由专家。

---

## 5. 补充说明与注意事项

### 5.1 关键维度避坑指南
1. **`num_key_value_heads`**: Kimi2 采用 GQA。在配置转换脚本时，**必须传入真实的 KV 头数（32）**，而不能传入 GQA 分组比例（2），否则会引发形状错位。
2. **LayerNorm 维度**: 注意力层附加了独立的 Q/K LayerNorm，其维度等于 `qk_nope_head_dim + qk_pos_emb_head_dim` (128+64=192)。

### 5.2 并行策略 (Parallelism) 影响边界
*   **TP (Tensor Parallel)**: 跨卡切分注意力矩阵和 FFN，转换时需要执行 `torch.cat` 或 `torch.chunk`。
*   **PP (Pipeline Parallel)**: 按层分配模型，转换时影响文件的遍历顺序。
*   **EP (Expert Parallel)**: 仅切割 MoE 层的专家，不影响 Dense 参数。
*   **VPP (Virtual Pipeline Parallel / DualPipe)**: Kimi2 使用 `dualpipev` 调度。这会导致层在 PP rank 内的分布非连续（呈 V 字型分布），转换脚本已通过 `_build_vpprank_layer_map` 实现了该非线性层映射的还原。

### 5.3 闭环一致性测试
项目中包含自动化测试脚本 `tests/weights/test_align_pretrain_config.py`，用于验证：
1. 配置解析的准确性。
2. `MCore -> HF -> MCore` 的端到端闭环。
3. 张量数据在切分和重组过程中的 SHA256 指纹绝对一致，确保零精度损失。