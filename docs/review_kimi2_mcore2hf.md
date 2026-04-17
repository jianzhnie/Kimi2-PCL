# Kimi2 MCore → HF 权重转换 Review 报告

> 对比文件: `utils/convert_kimi2_mcore2hf.py` vs `utils/convert_ckpt_deepseek3_mcore2hf.py`
> 分析日期: 2026-04-17

---

## 1. 整体架构差异 (设计差异)

| 特性 | DeepSeek3 | Kimi2 | 说明 |
|------|-----------|-------|------|
| 注意力机制 | **MLA** (Multi-head Latent Attention) | **GQA** (Grouped Query Attention) | 核心架构差异 |
| 专家数量 | 256 | 128 | 模型配置差异 |
| MTP 层 | 支持 (`mtp_flag`, `mtp.layers.*`) | 不需要 | Kimi2 无 MTP |
| LoRA | 支持 (`_merge_lora`, `save_lora_to_hf`) | 不需要 | Kimi2 不涉及 |
| `mla_mm_split` | 支持 | 不需要 | MLA 专用 |
| `expert_tp_size` | 无 (等同于 `expert_tp_size = tp_size`) | 支持 (默认 `expert_tp_size=1`) | **关键差异** |
| `torch_npu` | 导入 (昇腾 NPU) | 不导入 | CPU 环境兼容 |

---

## 2. Attention 转换逻辑对比

### DeepSeek3: MLA

MLA 将 KV 压缩到一个低秩潜在向量，需要两组投影：

```
MCore 存储:
  linear_qkv.weight     → [Q_LORA_RANK + KV_LORA_RANK, hidden]   (压缩的 Q/KV)
  linear_q_up_proj      → [num_heads * (qk_dim + rope_dim), Q_LORA_RANK]
  linear_kv_up_proj     → [num_heads * (qk_dim + v_dim), KV_LORA_RANK]
  kv_layernorm          → (压缩后的 KV 归一化)

HF 输出:
  q_a_proj              → [Q_LORA_RANK, hidden]
  kv_a_proj_with_mqa    → [KV_LORA_RANK, hidden]
  q_b_proj              → [num_heads * (qk_dim + rope_dim), Q_LORA_RANK]
  kv_b_proj             → [num_heads * (qk_dim + v_dim), KV_LORA_RANK]
```

### Kimi2: GQA

GQA 使用标准的 Q/K/V 线性投影，不做低秩压缩：

```
MCore 存储 (per TP rank):
  linear_qkv.weight     → [Q_tp + K_tp + V_tp, hidden]
                          布局: [Q_tp; K_tp; V_tp] along dim=0
  linear_proj.weight    → [hidden, O_tp]
  q_layernorm.weight    → (Q 归一化, 可选)
  k_layernorm.weight    → (K 归一化, 可选)

其中:
  Q_tp = (num_attention_heads / tp_size) * head_dim
  K_tp = (num_query_groups / tp_size) * head_dim
  V_tp = (num_query_groups / tp_size) * head_dim
```

### Kimi2 实现逻辑 (set_model_attn)

```python
# 步骤 1: 对每个 TP rank, 从 linear_qkv 中分离 Q/K/V
for tp_rank in self.tp_rank_list:
    cur_qkv = pop(qkv_key)                          # [Q_tp + K_tp + V_tp, hidden]
    q_r, k_r, v_r = torch.split(cur_qkv,
                                [q_per_tp, k_per_tp, v_per_tp], dim=0)
    q_parts.append(q_r)                              # [Q_tp, hidden]
    k_parts.append(k_r)                              # [K_tp, hidden]
    v_parts.append(v_r)                              # [V_tp, hidden]

# 步骤 2: 合并各 TP shard 的 Q/K/V
q_proj = torch.cat(q_parts, dim=0)   # [num_heads * head_dim, hidden]
k_proj = torch.cat(k_parts, dim=0)   # [num_groups * head_dim, hidden]
v_proj = torch.cat(v_parts, dim=0)   # [num_groups * head_dim, hidden]
o_proj = torch.cat(linear_proj_list, dim=1)  # [hidden, num_heads * head_dim]
```

**Round-trip 验证**:

hf2mcore 中: `qkv_shards[i] = cat([q_tp[i], k_tp[i], v_tp[i]], dim=0)`
mcore2hf 中: `split(qkv, [q_per_tp, k_per_tp, v_per_tp], dim=0)` → `cat(parts, dim=0)`

`split` 和 `chunk` 互为逆操作。✅

---

## 3. Dense MLP 转换逻辑

两个实现完全一致:

```python
# mcore2hf:
cur_fc1 = pop(fc1_key)                          # [2*ffn/tp, hidden]
cur_gate, cur_up = chunk(cur_fc1, 2, dim=0)     # 各 [ffn/tp, hidden]
gate_weights = cat(gate_list, dim=0)            # [ffn_hidden, hidden]
up_weights = cat(up_list, dim=0)                # [ffn_hidden, hidden]
down_weights = cat(down_list, dim=1)            # [hidden, ffn_hidden]
```

---

## 4. MoE 转换逻辑 — **核心差异**

### 4.1 Shared Experts — 一致 ✅

### 4.2 Router Bias — 可选处理

Kimi2 用 `.pop(router_bias_key, None)` 可选读取, 配合 hf2mcore 的可选写入。✅

### 4.3 Grouped GEMM + expert_tp_size — 核心差异

**DeepSeek3**: 始终 cat 所有 TP shard (等价于 expert_tp_size = tp_size)
**Kimi2 (expert_tp_size=1)**: 只取 tp_rank=0 的数据 (所有 TP rank 持有相同权重)

如果对 expert_tp_size=1 使用 cat-all-TP:
```
cat(TP shards, dim=2): [H, 2*I * tp]  ← 中间维度被扩大 tp 倍! 错误!
```

**Kimi2 (expert_tp_size>1)**: 选择 unique shard 的代表并 cat。

### Round-trip 验证 (expert_tp_size=1):

```
hf2mcore: gate=[12288,7168], up=[12288,7168]
  fc1 = cat([gate, up]) = [24576,7168], fc1.t() = [7168,24576]

mcore2hf: w1 = [7168, num_local*24576]
  reshape → [num_local, 7168, 24576]
  expert_i: reshape(7168,-1).t() = [24576,7168]
  chunk(2) → gate=[12288,7168], up=[12288,7168]  ✓
```

### 4.4 发现并修复的 Bug

#### Bug 1: unique TP 索引计算错误 (已修复)

**位置**: grouped_gemm 路径, expert_tp_size > 1 时

**问题**: `unique_tp_indices = [i * step for i in range(expert_tp_size)]`
在 tp_size=4, expert_tp_size=2 时选择 [0, 2], 但 TP 0 和 TP 2 持有**相同** shard。

**修复**: `unique_tp_indices = list(range(self.expert_tp_size))` → 选择 [0, 1]

**影响**: 仅 expert_tp_size > 1 时触发。默认配置不受影响。

#### Bug 2: 非 grouped_gemm 路径未处理 expert_tp_size > 1 (已修复)

**位置**: local_experts 格式, moe_tp_extend_ep=False 路径

**问题**: 仅从 tp_rank=0 读取数据, expert_tp_size > 1 时丢失其他 shard。

**修复**: 添加 expert_tp_size > 1 分支, 从前 expert_tp_size 个 TP rank 收集并拼接。

#### Bug 3: moe_tp_extend_ep=True 时 EP 目录索引和专家数量错误 (已修复)

**位置**: `__init__`, `get_pt_path_by_tpppep_rank`, MoE 专家转换逻辑

**问题**: `moe_tp_extend_ep=True` 时存在三个子问题:

1. **ep_rank_list 错误**: 使用 `range(ep_size)` = `range(64)`, 但实际目录结构中每个 TP rank 只有 `ep_size/tp_size = 32` 个 EP 目录。
   代码尝试加载 `mp_rank_00_000_001` (不存在), 导致 `FileNotFoundError`。

2. **路径构建错误**: 直接使用 `ep_rank` 作为目录名中的 EP 索引, 但实际目录使用交错命名:
   - TP=0: EP 索引为 0, 2, 4, ..., 62 (偶数)
   - TP=1: EP 索引为 1, 3, 5, ..., 63 (奇数)
   - 全局 EP 索引 = `tp_rank + ep_rank * tp_size`

3. **local_expert_nums 错误**: 使用 `num_experts // (tp_size * ep_size)` = 128/128 = 1, 但实际每个目录持有 `num_experts // ep_size` = 128/64 = 2 个专家。

**修复**:

```python
# 1. ep_rank_list: 每个 TP rank 只处理 ep_size/tp_size 个本地 EP
if self.moe_tp_extend_ep and self.tp_size > 1:
    self.ep_rank_list = list(range(self.ep_size // self.tp_size))  # range(32)

# 2. 路径: 使用全局 EP 索引 (交错命名)
global_ep = tp_rank + ep_rank * self.tp_size

# 3. local_expert_nums: 统一使用 num_experts // ep_size
local_expert_nums = self.num_experts // self.ep_size  # = 2
```

**验证** (EP=64, TP=2, 128 experts):

```
(tp=0, ep=0) → global_ep=0 → mp_rank_00_000_000 → experts 0,1
(tp=1, ep=0) → global_ep=1 → mp_rank_01_000_001 → experts 2,3
(tp=0, ep=1) → global_ep=2 → mp_rank_00_000_002 → experts 4,5
...
(tp=1, ep=31) → global_ep=63 → mp_rank_01_000_063 → experts 126,127
总计: 32 × 2 × 2 = 128 experts ✅
```

**影响**: `moe_tp_extend_ep=True` 时必然触发。这是默认训练配置, 属于**严重 bug**。

---

## 5. 汇总

| 转换路径 | 状态 | 说明 |
|---------|------|------|
| Embedding | ✅ | 一致 |
| Final Norm + LM Head | ✅ | Kimi2 无 MTP, 简化 |
| Layer Norm | ✅ | 一致 |
| Attention (GQA) | ✅ | 架构差异, GQA 实现正确 |
| Dense MLP | ✅ | 一致 |
| MoE Router | ✅ | 可选 bias, 正确 |
| MoE Shared Experts | ✅ | 一致 |
| MoE Grouped GEMM + tp_extend_ep | ✅ (已修复) | 修复 EP 目录索引和专家数量 |
| MoE Grouped GEMM + expert_tp_size=1 | ✅ | 正确: 不 cat TP shard |
| MoE Grouped GEMM + expert_tp_size>1 | ✅ (已修复) | 修复 unique TP 索引 |
| MoE Non-Grouped + tp_extend_ep | ✅ (已修复) | 修复 EP 目录索引和专家数量 |
| MoE Non-Grouped + expert_tp_size=1 | ✅ | 正确: 只读 tp=0 |
| MoE Non-Grouped + expert_tp_size>1 | ✅ (已修复) | 添加 gather 逻辑 |

**所有转换逻辑经过逐步数据流验证，与 hf2mcore 构成正确的逆操作。修复了 3 个 bug, 其中 Bug 3 在默认训练配置 (moe_tp_extend_ep=True) 下必然触发，属于严重 bug。**
