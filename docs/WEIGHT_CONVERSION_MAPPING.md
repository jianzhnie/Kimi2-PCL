# Kimi2-PCL 权重转换映射与验证文档

本文档系统性地梳理了 Kimi2-PCL 项目中 Kimi2-1T 模型在 Megatron-Core 与 Hugging Face 格式之间的权重转换规则、参数映射关系以及架构一致性验证细节。

## 1. 模型架构与参数配置

### 1.1 预训练核心参数 (Kimi2-1T)

基于 `pretrain_kimi2_1t_4k.sh` 脚本提取的基准模型架构参数：

| 参数模块       | 参数名称                  | 设定值     | 架构说明                          |
| :------------- | :------------------------ | :--------- | :-------------------------------- |
| **基础规模**   | `NUM_LAYERS`              | 32         | Transformer 总层数                |
|                | `HIDDEN_SIZE`             | 7168       | 隐藏层维度                        |
|                | `VOCAB_SIZE`              | 163840     | 词表大小                          |
|                | `MAX_POSITION_EMBEDDINGS` | 131072     | 最大支持上下文长度                |
| **注意力机制** | `NUM_ATTENTION_HEADS`     | 64         | Query 头数                        |
|                | `NUM_KEY_VALUE_HEADS`     | 32         | KV 头总数 (GQA)                   |
|                | `NUM_QUERY_GROUPS`        | 2          | GQA 分组比 (64/32=2)              |
|                | `QK_NOPE_HEAD_DIM`        | 128        | QK 无位置编码维度                 |
|                | `QK_ROPE_HEAD_DIM`        | 64         | QK 位置编码 (RoPE) 维度           |
|                | `Q_HEAD_DIM` (总)         | **192**    | 128 + 64，实际 Q/K 头维度         |
|                | `V_HEAD_DIM`              | 128        | V 头维度                          |
|                | `QK_LAYERNORM`            | True       | 启用 QK 独立 LayerNorm            |
| **MoE 架构**   | `NUM_EXPERTS`             | 128        | 路由专家总数                      |
|                | `N_SHARED_EXPERTS`        | 1          | 共享专家数量                      |
|                | `FIRST_K_DENSE_REPLACE`   | 2          | 前 2 层使用 Dense MLP，不使用 MoE |
|                | `MOE_FFN_HIDDEN_SIZE`     | 12288      | 专家网络的中间层维度              |
|                | `FFN_HIDDEN_SIZE`         | 18432      | Dense 层的中间层维度              |
|                | `MOE_ROUTER_TOPK`         | 2          | 路由 Top-K 激活数                 |
| **位置编码**   | `ROTARY_BASE`             | 50000      | RoPE 基础频率                     |
|                | `ROPE_SCALING_TYPE`       | yarn       | RoPE 缩放类型 (Factor: 32)        |
| **并行策略**   | `TP` / `PP` / `EP`        | 2 / 8 / 64 | 张量并行 / 流水线并行 / 专家并行  |
|                | `SCHEDULES_METHOD`        | dualpipev  | 双向流水线调度策略                |

> **注意**: HIDDEN_SIZE (7168) / NUM_ATTENTION_HEADS (64) = 112，但 Kimi2-1T 使用 **Decoupled Head Dimensions** 架构，实际的 `Q_HEAD_DIM = 192` (128+64)，而不是 112。这是通过独立指定 `qk_nope_head_dim` 和 `qk_rope_head_dim` 实现的。

### 1.2 参数命名映射表

不同组件间的参数命名对应关系：

| 概念              | HF Config             | 转换脚本参数            | 训练脚本参数                                                       |
| :---------------- | :-------------------- | :---------------------- | :----------------------------------------------------------------- |
| QK 无位置编码维度 | `qk_nope_head_dim`    | `--qk-head-dim`         | `--kv-channels`                                                    |
| QK 位置编码维度   | `qk_rope_head_dim`    | `--qk-pos-emb-head-dim` | (使用默认值 64)                                                    |
| V 头维度          | `v_head_dim`          | `--v-head-dim`          | (使用默认值 128)                                                   |
| 隐藏层维度        | `hidden_size`         | `--hidden-size`         | `--hidden-size`                                                    |
| KV 头总数         | `num_key_value_heads` | `--num-key-value-heads` | `--num-query-groups` (注意：这是分组比，实际 KV heads = 64/2 = 32) |

### 1.3 转换脚本参数对齐状态

转换工具 (`convert_ckpt_hf2mcore.py` / `convert_ckpt_mcore2hf.py`) 的默认配置与预训练脚本的一致性：

| 核心参数                | 转换脚本设定                             | 预训练脚本设定      | 对齐状态 |
| :---------------------- | :--------------------------------------- | :------------------ | :------: |
| `HIDDEN_SIZE`           | 7168                                     | 7168                |    ✅     |
| `NUM_EXPERTS`           | 128                                      | 128                 |    ✅     |
| `FIRST_K_DENSE_REPLACE` | 2                                        | 2                   |    ✅     |
| `NUM_ATTENTION_HEADS`   | 64                                       | 64                  |    ✅     |
| `NUM_KEY_VALUE_HEADS`   | 32                                       | 32 (64/2)           |    ✅     |
| `QK_NOPE_HEAD_DIM`      | 128                                      | 128 (--kv-channels) |    ✅     |
| `QK_ROPE_HEAD_DIM`      | 64                                       | 64 (默认)           |    ✅     |
| `V_HEAD_DIM`            | 128                                      | 128 (默认)          |    ✅     |
| `VOCAB_SIZE`            | 163840                                   | 163840              |    ✅     |
| `ROTARY_BASE`           | 50000                                    | 50000               |    ✅     |
| `MOE_GROUPED_GEMM`      | 启用                                     | 启用                |    ✅     |
| `SCHEDULES_METHOD`      | dualpipev                                | dualpipev           |    ✅     |
| `QK_LAYERNORM`          | HF→MCore: 默认启用<br>MCore→HF: 默认启用 | 启用                |    ✅     |

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

| Megatron-Core (MCore)                                 | Hugging Face (HF)                                                                                                                      | 形状变换与操作                                                                                       |
| :---------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| `decoder.layers.{i}.self_attention.linear_qkv.weight` | `model.layers.{i}.self_attn.q_proj.weight`<br>`model.layers.{i}.self_attn.k_proj.weight`<br>`model.layers.{i}.self_attn.v_proj.weight` | **MCore→HF**: 沿 dim=0 切分为 Q/K/V，再按 TP 合并<br>**HF→MCore**: 沿 dim=0 按 TP 切分，再拼接 Q+K+V |

### 2.3 Dense MLP 层 (前 `first_k_dense_replace` 层)

| Megatron-Core (MCore)                      | Hugging Face (HF)                                                                | 形状变换与操作         |
| :----------------------------------------- | :------------------------------------------------------------------------------- | :--------------------- |
| `decoder.layers.{i}.mlp.linear_fc1.weight` | `model.layers.{i}.mlp.gate_proj.weight`<br>`model.layers.{i}.mlp.up_proj.weight` | 沿 dim=0 对半切分/拼接 |
| `decoder.layers.{i}.mlp.linear_fc2.weight` | `model.layers.{i}.mlp.down_proj.weight`                                          | TP 合并/分割 (dim=1)   |

### 2.4 MoE 专家层 (后续层)

| Megatron-Core (MCore) | Hugging Face (HF) | 形状变换与操作 |
| :-------------------- | :---------------- | :------------- |
| `decoder.layers.{i}.mlp.router.weight` | `model.layers.{i}.mlp.gate.weight` | 跨 EP 合并/分割 |
| `decoder.layers.{i}.mlp.router.expert_bias` | `model.layers.{i}.mlp.gate.e_score_correction_bias` | 跨 EP 合并/分割 |
| `decoder.layers.{i}.mlp.shared_experts.linear_fc1.weight` | `model.layers.{i}.mlp.shared_experts.gate_proj.weight`<br>`model.layers.{i}.mlp.shared_experts.up_proj.weight` | 切分 gate/up + TP 变换 |
| `decoder.layers.{i}.mlp.shared_experts.linear_fc2.weight` | `model.layers.{i}.mlp.shared_experts.down_proj.weight` | TP 变换 |
| `decoder.layers.{i}.mlp.experts.weight1` (Grouped) | `model.layers.{i}.mlp.experts.{e}.gate_proj.weight`<br>`model.layers.{i}.mlp.experts.{e}.up_proj.weight` | 见 3.2 专家重组逻辑 |
| `decoder.layers.{i}.mlp.experts.weight2` (Grouped) | `model.layers.{i}.mlp.experts.{e}.down_proj.weight` | 见 3.2 专家重组逻辑 |

---

## 3. 核心张量变换逻辑详解

### 3.1 QKV 分离与合并逻辑

Megatron-Core 采用连续内存存储 QKV，而 Hugging Face 将其独立为三个投影矩阵。在 Kimi2-1T (GQA) 架构下：

*   **Q_head_dim** = `qk_nope_head_dim` (128) + `qk_rope_head_dim` (64) = **192**
*   **K_head_dim** = `qk_nope_head_dim` (128) + `qk_rope_head_dim` (64) = **192**
*   **V_head_dim** = 128

> 注意：这是 **Decoupled Head Dimensions** 架构，head_dim 不由 `hidden_size / num_heads` (7168/64=112) 决定，而是独立指定。

**MCore → HF (提取与合并)**
```python
# 1. 计算每个 TP Rank 内的头部数量
heads_per_tp = num_attention_heads // tp_size           # 64 / 2 = 32
kv_heads_per_tp = num_key_value_heads // tp_size        # 32 / 2 = 16

# 2. 计算每个 TP Rank 内张量的行数
q_per_tp = heads_per_tp * q_head_dim                    # 32 * 192 = 6144
k_per_tp = kv_heads_per_tp * q_head_dim                 # 16 * 192 = 3072
v_per_tp = kv_heads_per_tp * v_head_dim                 # 16 * 128 = 2048

# 3. 切分单个 TP 的 QKV
q_r, k_r, v_r = torch.split(qkv_shard, [q_per_tp, k_per_tp, v_per_tp], dim=0)

# 4. 合并所有 TP 分片
q_proj = torch.cat(q_parts, dim=0)  # 最终 [64 * 192, 7168] = [12288, 7168]
k_proj = torch.cat(k_parts, dim=0)  # 最终 [32 * 192, 7168] = [6144, 7168]
v_proj = torch.cat(v_parts, dim=0)  # 最终 [32 * 128, 7168] = [4096, 7168]
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
| **Attention 层**      | `32 * [ (64*192 + 32*192 + 32*128 + 64*192) * 7168 ]` | ~8.39 B       |
| **Dense MLP (前2层)** | `2 * [ 3 * 7168 * 18432 ]`                            | ~0.79 B       |
| **MoE Shared (30层)** | `30 * [ 3 * 7168 * 12288 ]`                           | ~7.92 B       |
| **MoE Routed (30层)** | `30 * 128 * [ 3 * 7168 * 12288 ]`                     | ~1,014.6 B    |
| **总计估算**          |                                                       | **~1.03 T**   |

> 注意：Attention 层计算修正为使用实际的 head_dim (192) 而非传统计算的 112。

### 4.2 检查点分片体积 (TP=2, PP=8, EP=64)

在分布式训练/转换中，单张卡的显存占用由以下切分决定：

*   **PP 切分**: 每卡负责 `32 / 8 = 4` 层。
*   **TP 切分**: 词嵌入、输出层、Attention QKV/O、MLP 参数均被分为 2 份。
*   **EP 切分**: 每卡仅负责 `128 / 64 = 2` 个路由专家。

---

## 5. 已知问题与注意事项

### 5.1 关键维度避坑指南

1. **`num_key_value_heads` 语义**: Kimi2 采用 GQA。在配置转换脚本时，**必须传入真实的 KV 头总数（32）**，而不能传入 GQA 分组比例（2），否则会引发形状错位。

2. **`--kv-channels` 与 `--qk-head-dim` 对应关系**: 训练脚本使用 `--kv-channels` (Megatron 命名)，转换脚本使用 `--qk-head-dim` (HF 命名)，两者都等于 `qk_nope_head_dim` (128)。

3. **head_dim 计算**: 实际的 Q/K head_dim = 192 (128+64)，不是 `hidden_size / num_heads` = 112。这是 Decoupled Head Dimensions 架构的特性。

4. **LayerNorm 维度**: 注意力层附加了独立的 Q/K LayerNorm，其维度等于 `qk_nope_head_dim + qk_rope_head_dim` (128+64=192)。

### 5.2 当前存在的问题

#### 问题 1: QK_LAYERNORM 在 MCore→HF 转换中缺失 ✅ 已修复

| 组件                                               | QK_LAYERNORM 支持           |
| :------------------------------------------------- | :-------------------------- |
| 训练脚本 (`pretrain_kimi2_1t_4k.sh`)               | ✅ `--qk-layernorm`          |
| HF→MCore 转换脚本 (`ckpt_convert_hf2mcore.sh`)     | ✅ 默认启用 `--qk-layernorm` |
| MCore→HF 转换脚本 (`ckpt_convert_mcore2hf.sh`)     | ✅ 默认启用 `--qk-layernorm` |
| MCore→HF 转换代码 (`convert_ckpt_mcore2hf.py`)     | ✅ 支持 `--qk-layernorm`     |

**修复内容**:
1. 在 `convert_ckpt_mcore2hf.py` 中添加 `--qk-layernorm` 参数
2. 在 `ckpt_convert_mcore2hf.sh` 中默认启用 QK_LAYERNORM
3. 转换后的 HF config 正确写入 `"qk_layernorm": true`

**验证方式**:
```bash
# 检查生成的 config.json 是否包含 qk_layernorm
cat ${SAVE_DIR}/config.json | grep qk_layernorm
# 预期输出: "qk_layernorm": true
```

#### 问题 2: MOE_TP_EXTEND_EP 设置不一致 ⭐ 需要关注

| 组件              | MOE_TP_EXTEND_EP              | 验证逻辑 |
| :---------------- | :---------------------------- | :------- |
| 训练脚本          | 未明确设置 (默认可能为 0)     | 无       |
| HF→MCore 转换脚本 | **默认启用 (1)**              | 有       |
| MCore→HF 转换脚本 | **默认禁用 (0)**              | 无       |

**影响**: 双向转换策略不一致，可能导致权重分布差异

**建议**: 统一默认值，或明确文档说明使用场景
```bash
# 当前差异
# HF→MCore: export MOE_TP_EXTEND_EP="${MOE_TP_EXTEND_EP:-1}"  # 默认启用
# MCore→HF: if [[ "${MOE_TP_EXTEND_EP:-0}" == "1" ]]          # 默认禁用

# 建议统一为默认禁用
export MOE_TP_EXTEND_EP="${MOE_TP_EXTEND_EP:-0}"
```

#### 问题 3: 双向转换脚本参数对比

**对比对象**: `ckpt_convert_hf2mcore.sh` vs `ckpt_convert_mcore2hf.sh`

| 对比项 | HF→MCore | MCore→HF | 状态 |
| :----- | :-------- | :-------- | :--- |
| **核心参数** | | | |
| TP/PP/EP | 2/8/64 | 2/8/64 | ✅ |
| NUM_LAYERS | 32 | 32 | ✅ |
| HIDDEN_SIZE | 7168 | 7168 | ✅ |
| NUM_EXPERTS | 128 | 128 | ✅ |
| QK_LAYERNORM | 默认启用 | 默认启用 | ✅ |
| **参数命名** | | | |
| TP 参数名 | `--target-tensor-parallel-size` | `--source-tensor-parallel-size` | N/A |
| PP 参数名 | `--target-pipeline-parallel-size` | `--source-pipeline-parallel-size` | N/A |
| EP 参数名 | `--target-expert-parallel-size` | `--source-expert-parallel-size` | N/A |
| **默认值差异** | | | |
| IO_THREADS | 2 | 8 | ⚠️ |
| MOE_TP_EXTEND_EP | 1 (启用) | 0 (禁用) | ❌ |
| **特有参数** | | | |
| SAVE_WORKERS | 有 (并行保存) | 无 | N/A |
| config.json 检查 | 有 | 无 | N/A |

**IO_THREADS 差异说明**:
- HF→MCore: 使用 2 个线程读取 HF 权重
- MCore→HF: 使用 8 个线程读取 MCore 权重
- 这是根据转换方向优化的，可接受

**SAVE_WORKERS 说明**:
- 仅 HF→MCore 需要，用于并行保存 MCore 格式的多文件权重
- MCore→HF 保存为单文件 HF 格式，不需要并行保存

#### 问题 4: ROPE 缩放参数未传递

训练脚本配置了完整的 Yarn RoPE 参数 (`--rope-scaling-type yarn --rope-scaling-factor 32 ...`)，但转换脚本仅传递 `--rotary-base 50000`。

**影响**: 长上下文 (>4096) 的位置编码可能不一致。

**建议**: 确认是否需要将这些参数写入 HF config 的 `rope_scaling` 字段。

### 5.3 并行策略 (Parallelism) 影响边界

*   **TP (Tensor Parallel)**: 跨卡切分注意力矩阵和 FFN，转换时需要执行 `torch.cat` 或 `torch.chunk`。
*   **PP (Pipeline Parallel)**: 按层分配模型，转换时影响文件的遍历顺序。
*   **EP (Expert Parallel)**: 仅切割 MoE 层的专家，不影响 Dense 参数。
*   **VPP (Virtual Pipeline Parallel / DualPipe)**: Kimi2 使用 `dualpipev` 调度。这会导致层在 PP rank 内的分布非连续（呈 V 字型分布），转换脚本已通过 `_build_vpprank_layer_map` 实现了该非线性层映射的还原。

### 5.4 闭环一致性测试

项目中包含自动化测试脚本，用于验证转换的正确性：

| 测试脚本                                 | 用途                       |
| :--------------------------------------- | :------------------------- |
| `tests/test_align_pretrain_config.py`    | 验证配置解析的准确性       |
| `tests/test_conversion_comprehensive.py` | 验证 MCore ↔ HF 双向转换   |
| `tests/test_parallel_saving.py`          | 验证并行保存功能           |
| `utils/check_model_weights.py`           | 验证权重完整性和参数量统计 |

**端到端闭环测试流程**:
```bash
# 1. MCore → HF
LOAD_DIR=/path/to/mcore/ckpt SAVE_DIR=/path/to/hf/output scripts/ckpt_convert_mcore2hf.sh

# 2. HF → MCore
LOAD_DIR=/path/to/hf/output SAVE_DIR=/path/to/mcore/output scripts/ckpt_convert_hf2mcore.sh

# 3. 对比验证
python utils/check_model_weights.py --original /path/to/mcore/ckpt --converted /path/to/mcore/output
```

---

## 6. 附录

### 6.1 配置文件示例 (`config_1t.json`)

```json
{
    "hidden_size": 7168,
    "num_attention_heads": 64,
    "num_key_value_heads": 32,
    "num_query_groups": 2,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "qk_layernorm": true,
    "rope_scaling": {
        "type": "yarn",
        "factor": 32.0,
        "beta_fast": 1.0,
        "beta_slow": 1.0,
        "original_max_position_embeddings": 4096
    }
}
```

### 6.2 参考文档

- [ARCHITECTURE_DIFFERENCE.md](ARCHITECTURE_DIFFERENCE.md) - 架构差异说明
- [tests/README.md](../tests/README.md) - 测试文档
