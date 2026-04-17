# Kimi2 HF → MCore 转换代码 Review

> 对比文件: `utils/convert_kimi2_hf2mcore.py` vs `utils/convert_ckpt_deepseek3.py`
> 启动脚本: `scripts/ckpt_convert_kimi2_hf2mcore.sh`
> 训练配置: `scripts/pretrain_kimi2_1t_4k.sh`
> 分析日期: 2026-04-17

---

## 1. 总览

| 项目 | 说明 |
|------|------|
| 转换脚本 | `utils/convert_kimi2_hf2mcore.py` |
| 启动脚本 | `scripts/ckpt_convert_kimi2_hf2mcore.sh` |
| 参考实现 | `utils/convert_ckpt_deepseek3.py` (DeepSeek-V3, MLA+MoE) |
| 架构差异 | **GQA+MoE** (非 MLA)，支持 QK LayerNorm，无 MTP 层 |

---

## 2. Shell 脚本 Review (`ckpt_convert_kimi2_hf2mcore.sh`)

### 2.1 默认参数与训练脚本一致性校验

| 参数 | 转换脚本默认值 | 训练脚本值 | 匹配 |
|------|---------------|-----------|------|
| NUM_LAYERS | 32 | 32 | ✓ |
| HIDDEN_SIZE | 7168 | 7168 | ✓ |
| NUM_ATTENTION_HEADS | 64 | 64 | ✓ |
| NUM_QUERY_GROUPS | 2 | 2 | ✓ |
| KV_CHANNELS | 128 | 128 | ✓ |
| FFN_HIDDEN_SIZE | 18432 | 18432 | ✓ |
| MOE_FFN_HIDDEN_SIZE | 12288 | 12288 | ✓ |
| NUM_EXPERTS | 128 | 128 | ✓ |
| FIRST_K_DENSE_REPLACE | 2 | 2 | ✓ |
| VOCAB_SIZE | 163840 | 163840 | ✓ |
| QK_LAYERNORM | 1 (启用) | 1 (启用) | ✓ |
| MOE_GROUPED_GEMM | 1 (启用) | 启用 | ✓ |
| TP | 2 | 2 | ✓ |
| PP | 8 | 8 | ✓ |
| EP | 8 | 64 | ⚠ 默认不同，需按训练配置调整 |
| EXPERT_TP | 1 | 1 | ✓ |

> **说明**: EP 默认值 (8) 与训练脚本 (64) 不同，用户需要通过环境变量 `EP=64` 指定。这不是 bug，但建议在脚本注释中提醒。

### 2.2 三种并行模式支持

脚本正确支持三种并行模式：

1. **DualPipeV 模式** (`--schedules-method dualpipev`):
   - 训练脚本使用的模式
   - 正确传递 `--schedules-method` 参数
   - 与 `VPP_STAGE` 互斥检查 ✓

2. **标准 VPP 模式** (`--num-layers-per-virtual-pipeline-stage N`):
   - 通过 `VPP_STAGE` 环境变量控制
   - 正确传递参数 ✓

3. **纯 PP 模式**:
   - 不设置 `schedules-method` 和 `vpp-stage`
   - 默认模式 ✓

Shell 脚本逻辑正确，参数完整，三种模式切换合理。

---

## 3. 转换代码 Review (`convert_kimi2_hf2mcore.py`)

### 3.1 类初始化与参数校验 (`__init__`, `_valid_parameter`)

**校验规则清单**：

| 校验项 | 规则 | 正确性 |
|--------|------|--------|
| first_k_dense_replace | ∈ [0, num_layers] | ✓ |
| num_experts % ep_size | == 0 | ✓ |
| num_attention_heads % tp_size | == 0 | ✓ |
| num_query_groups % tp_size | == 0 | ✓ |
| expert_tp_size ≤ tp_size | 数值约束 | ✓ |
| tp_size % expert_tp_size | == 0 | ✓ |
| moe_tp_extend_ep vs expert_tp_size>1 | 互斥 | ✓ |
| dualpipe + tp>1 | 警告 expert_tp_size=1 | ✓ |
| num_layers % pp_size | == 0 | ✓ |
| num_layer_list vs vpp | 互斥 | ✓ |
| num_layer_list vs noop_layers | 互斥 | ✓ |

**与 DeepSeek3 差异**：

- DeepSeek3 强制 `dualpipe + tp>1` 时必须启用 `moe_tp_extend_ep`（raise ValueError）
- Kimi2 放宽为警告（logger.warning），因为 expert_tp_size=1 已经是常见配置
- DeepSeek3 限制 `first_k_dense_replace ≤ 3`，Kimi2 允许 [0, num_layers] 范围（更灵活）

### 3.2 层映射逻辑

#### 3.2.1 Pure PP 模式 (`get_pprank_hf_layeridxs`)

以 32 层、PP=8 为例：

```
pp_rank=0: [0, 1, 2, 3]     ← 前 2 层为 Dense (layer 0, 1)
pp_rank=1: [4, 5, 6, 7]
pp_rank=2: [8, 9, 10, 11]
...
pp_rank=7: [28, 29, 30, 31]
```

顺序映射，逻辑正确。支持 noop_layers 的正确扣除。

#### 3.2.2 DualPipe 模式 (`get_vpprank_hf_layeridxs`)

DualPipe 将每 PP stage 的层分为前后两半，分别分配给 vpp_rank=0 和 vpp_rank=1：

以 32 层、PP=8、vpp_size=2 为例，每 PP stage 有 4 层：

```
dualpipe_layer_list 构建过程:
  all_layer_list = [0,1,2,...,31]
  layer_pop_num = 32/8/2 = 2

  迭代1: 取前2 [0,1] + 取后2 [30,31] → all = [2,...,29]
  迭代2: 取前2 [2,3] + 取后2 [28,29] → all = [4,...,27]
  ...
  迭代8: 取前2 [14,15] + 取后2 [16,17] → all = []

结果:
  pp_rank=0: vpp0=[0,1]   vpp1=[30,31]
  pp_rank=1: vpp0=[2,3]   vpp1=[28,29]
  pp_rank=2: vpp0=[4,5]   vpp1=[26,27]
  ...
  pp_rank=7: vpp0=[14,15] vpp1=[16,17]
```

与 DeepSeek3 的 `dualpipe_layer_list` 构建逻辑完全一致，实现正确。

#### 3.2.3 load_matched_hf_weights

权重加载逻辑：

| 条件 | 加载内容 |
|------|---------|
| pp_rank=0, 非 dualpipe | embed_tokens + 当前 PP 层 |
| pp_rank=0, dualpipe | embed_tokens + lm_head + model.norm + 当前 PP 层 |
| pp_rank=last, 非 dualpipe | lm_head + model.norm + 当前 PP 层 |

与 DeepSeek3 一致。DualPipe 模式下 embed 和 lm_head 都在 pp_rank=0，正确。

### 3.3 Attention 层转换 (`set_model_layer_attn`)

#### 3.3.1 GQA 权重处理

Kimi2 采用 GQA 架构，与 DeepSeek3 的 MLA 有本质区别：

**Kimi2 (GQA) 权重来源**：
```
q_proj.weight:  [num_attention_heads * qk_head_dim, hidden] = [8192, 7168]
k_proj.weight:  [num_query_groups * qk_head_dim, hidden]    = [256, 7168]
v_proj.weight:  [num_query_groups * v_head_dim, hidden]     = [256, 7168]
o_proj.weight:  [hidden, num_attention_heads * v_head_dim]  = [7168, 8192]
```

**TP 分片策略**：
```python
q_tp = torch.chunk(q_weight, tp_size, dim=0)   # Q 按 head 数切分
k_tp = torch.chunk(k_weight, tp_size, dim=0)   # K 按 group 数切分
v_tp = torch.chunk(v_weight, tp_size, dim=0)   # V 按 group 数切分
dense_tp = torch.chunk(dense_weight, tp_size, dim=1)  # O 按列切分
```

以 TP=2 为例，每个 TP shard 的 QKV：
```
q_shard: [4096, 7168]  (32 heads × 128 dim)
k_shard: [128, 7168]   (1 group × 128 dim)
v_shard: [128, 7168]   (1 group × 128 dim)
→ qkv_shard: [4352, 7168]   ← Megatron linear_qkv 标准布局
```

这是标准的 Megatron GQA QKV 布局，每个 TP rank 持有等比例的 Q heads 和对应的 KV groups。

**Shape 校验**：
```python
expected_q_rows = num_attention_heads * qk_head_dim  # 64 * 128 = 8192
expected_k_rows = num_query_groups * qk_head_dim     # 2 * 128 = 256
expected_v_rows = num_query_groups * v_head_dim       # 2 * 128 = 256
```
提供了维度校验，防止配置错误导致静默错误。✓

#### 3.3.2 QK LayerNorm 处理

```python
q_ln = weights_dict.pop(f"...self_attn.q_layernorm.weight", None)
k_ln = weights_dict.pop(f"...self_attn.k_layernorm.weight", None)
```

- Q 和 K 有独立的 LayerNorm（GQA 特有）
- 使用 `None` 默认值，兼容不带 QK LayerNorm 的模型
- LayerNorm 权重在 EP/TP ranks 间复制（不切分），正确 ✓
- 丢弃 `rotary_emb.inv_freq`（转换时不需要），正确 ✓

#### 3.3.3 与 DeepSeek3 对比

| 特性 | Kimi2 (GQA) | DeepSeek3 (MLA) |
|------|-------------|-----------------|
| Q/K/V 投影 | 独立的 q/k/v_proj | 压缩的 q_a_proj + kv_a_proj |
| 上投影 | 无 | q_b_proj + kv_b_proj |
| QKV TP 切分 | 按 head 切分后拼接 | 压缩部分不切，上投影按 TP 切 |
| LayerNorm | q_layernorm + k_layernorm | q_a_layernorm + kv_a_layernorm |
| dense_proj | 按 dim=1 切 | 按 dim=1 切 |
| mla_mm_split | 无 | 支持 (将 2 个 up_proj 拆为 4 个) |

两者都正确实现了各自的注意力机制。

### 3.4 MLP 层转换 (`set_model_layer_mlp`)

#### 3.4.1 Dense 层

```
gate_proj [ffn_hidden, hidden] = [18432, 7168]
up_proj   [ffn_hidden, hidden] = [18432, 7168]
down_proj [hidden, ffn_hidden] = [7168, 18432]
```

TP 切分：
```python
gate_chunks = chunk(gate_proj, tp_size, dim=0)   # [9216, 7168] × 2
up_chunks = chunk(up_proj, tp_size, dim=0)        # [9216, 7168] × 2
fc1 = [gate_i; up_i] for each TP shard            # [18432, 7168] × 2
fc2 = chunk(down_proj, tp_size, dim=1)             # [7168, 9216] × 2
```

SwiGLU 的 gate+up 交织后按 TP 切分，逻辑正确。与 DeepSeek3 一致。✓

#### 3.4.2 自动检测 Dense/MoE 层

Kimi2 增加了基于实际 HF checkpoint 键的自动检测逻辑：

```python
is_dense_layer = hf_layer_idx < self.first_k_dense_replace
has_moe_key = "mlp.gate.weight" in weights_dict
has_dense_key = "mlp.gate_proj.weight" in weights_dict

# 交叉校验：如果配置与实际键冲突，自动纠正并发出警告
if is_dense_layer and has_moe_key and not has_dense_key:
    is_dense_layer = False  # 自动切换到 MoE 路径
elif not is_dense_layer and has_dense_key and not has_moe_key:
    is_dense_layer = True   # 自动切换到 Dense 路径
elif not is_dense_layer and not has_moe_key and not has_dense_key:
    raise KeyError(...)     # 两种键都不存在，报错
```

这是一个很好的防御性编程实践，比 DeepSeek3 单纯依赖索引比较更鲁棒。✓

### 3.5 MoE Expert 权重处理

#### 3.5.1 单个 Expert 权重构建

```python
for expert_idx in range(num_experts):
    gate = [moe_ffn_hidden, hidden] = [12288, 7168]
    up   = [moe_ffn_hidden, hidden] = [12288, 7168]
    down = [hidden, moe_ffn_hidden] = [7168, 12288]

    gate_chunks = chunk(gate, expert_tp_size, dim=0)
    up_chunks = chunk(up, expert_tp_size, dim=0)
    fc1 = interleave(gate_chunks, up_chunks)  # [24576, 7168]
    fc1_transposed = fc1.t()                    # [7168, 24576]

    experts_linear_fc1_list.append(fc1_transposed)
    experts_linear_fc2_list.append(down.t())    # [12288, 7168]
```

**Gate/Up 交织模式**（以 expert_tp_size=2 为例）：
```
gate_chunks = [g0, g1]  各 [6144, 7168]
up_chunks   = [u0, u1]  各 [6144, 7168]

Kimi2:   [g0, u0, g1, u1]  →  [24576, 7168]  → .t() → [7168, 24576]
DeepSeek3: [g0;u0, g1;u1]  →  [24576, 7168]  → .t() → [7168, 24576]
```

两者产生相同的内存布局（gate 和 up 按 TP shard 配对），在 grouped_gemm 3D 视图下按 TP 切分时，每个 TP shard 均能获得正确的 (gate_i, up_i) 配对。等效实现。✓

#### 3.5.2 Grouped GEMM 路径

**Shape 推导**（默认配置：EP=8, TP=2, expert_tp_size=1, num_experts=128）：

```
Step 1: 每个 expert 的 fc1 = [7168, 24576], fc2 = [12288, 7168]

Step 2: 拼接
  gemm_fc1 = cat(128 × [7168, 24576]).view(7168, -1)
           = [917504, 24576].view(7168, 3145728)
           = [7168, 3145728]

Step 3: 3D reshape
  gemm_fc1_3d = [7168, 3145728].view(128, 7168, 24576)

Step 4: EP 切分
  gemm_fc1_ep = chunk([128, 7168, 24576], 8, dim=0)
              → 8 chunks of [16, 7168, 24576]
```

每个 EP rank 获得 16 个 expert 的权重，每个 expert shape = [7168, 24576]。

**expert_tp_size=1 时**（默认）：
```python
fc1_shards = [fc1_ep]  # 不切分，所有 TP rank 获得相同权重
fc2_shards = [fc2_ep]
# tp_rank % 1 == 0, 所有 TP rank 都取 shard[0]
```
→ 所有 TP rank 持有相同的 expert 权重副本（experts 不参与 TP 切分）。✓

**expert_tp_size>1 时**：
```python
fc1_shards = chunk(fc1_ep, expert_tp_size, dim=2)  # 按中间维度切
fc2_shards = chunk(fc2_ep, expert_tp_size, dim=1)   # 按中间维度切
# tp_rank % expert_tp_idx → 正确路由到对应 shard
```
→ 每个 TP shard 获得配对的 gate+up 子集。✓

**moe_tp_extend_ep 模式**：
```python
bucket_num = ep_size * tp_size  # 合并 EP 和 TP 作为 expert 分桶
gemm_fc1_ep = chunk(gemm_fc1_3d, bucket_num, dim=0)
idx = ep_rank * tp_size + tp_rank  # 线性映射
```
→ 每个 (ep_rank, tp_rank) 对获得独立的 expert 子集。✓

#### 3.5.3 Non-Grouped GEMM 路径

**fc1/fc2 双重转置分析**：

```python
# 构建: fc1.t() → [hidden, 2*mffhs] 存入 list
experts_linear_fc1_list.append(fc1.t())

# 使用: 再次 .t() 恢复原始 shape
local_fc1 = experts_linear_fc1_list[idx].t()  # [2*mffhs, hidden]
```

双重 `.t()` 恢复了 [out_features, in_features] 格式：
- fc1: [2*mffhs, hidden] = [24576, 7168] → Megatron ColumnParallelLinear 标准布局 ✓
- fc2: [hidden, mffhs] = [7168, 12288] → Megatron RowParallelLinear 标准布局 ✓

与 DeepSeek3 处理方式完全一致。✓

**expert_tp_size=1, tp_size=2 时**：
```python
fc1_shards = [local_fc1.clone()]  # 1 个 shard
for tp_rank in range(tp_size):
    expert_tp_idx = tp_rank % 1  # = 0
    mg_model[ep_rank][tp_rank][...] = fc1_shards[0]
```
→ 所有 TP rank 持有完整的 expert fc1/fc2 权重。正确，因为 expert 不参与 TP。✓

#### 3.5.4 Shared Expert 处理

```python
shared_gate_chunks = chunk(shared_gate, tp_size, dim=0)
shared_up_chunks = chunk(shared_up, tp_size, dim=0)
shared_fc1_shards = [cat([g, u]) for g, u in zip(gate_chunks, up_chunks)]
shared_fc2_shards = chunk(shared_down, tp_size, dim=1)
```

Shared expert 始终按 **tp_size** 切分（不受 expert_tp_size 影响），与 DeepSeek3 一致。✓

> **效率改进**：DeepSeek3 将 shared expert 的切分放在 expert 循环内部，导致重复计算 `num_experts` 次。Kimi2 正确地移到循环外。

### 3.6 Router 权重处理

```python
router_w = weights_dict.pop("mlp.gate.weight")
if router_w.shape[0] != self.num_experts:
    router_w = router_w[:self.num_experts, :].clone()

router_b = weights_dict.pop("mlp.gate.e_score_correction_bias", None)
```

- Router 权重仅在行数不匹配时截断（比 DeepSeek3 的无条件截断更优）
- Router bias 可选（DeepSeek3 为必选），对 Kimi2 更友好
- Router 权重在 EP/TP ranks 间复制（不切分），正确 ✓

### 3.7 Checkpoint 保存

#### 3.7.1 目录结构

```
save_dir/
├── latest_checkpointed_iteration.txt  (内容: "1")
└── iter_0000001/
    ├── mp_rank_00_00_000/  (tp=0, pp=0, ep=0)
    │   └── model_optim_rng.pt
    ├── mp_rank_01_00_000/  (tp=1, pp=0, ep=0)
    ...
```

命名规则与 DeepSeek3 一致。✓

#### 3.7.2 保存内容

```python
torch.save({
    'model': mg_model[ep_rank][tp_rank],
    'checkpoint_version': 3.0,
    'iteration': 1,
    'args': self._build_checkpoint_args(),  # ← Kimi2 新增
}, save_file_name, pickle_protocol=4, _use_new_zipfile_serialization=True)
```

**改进点**：Kimi2 增加了 `'args'` 字段，保存模型配置参数。这允许 Megatron 在加载时自动获取模型结构信息。DeepSeek3 没有此字段。

#### 3.7.3 `_build_checkpoint_args` 完整性校验

与训练脚本的配置逐项对比：

| Args 字段 | Kimi2 设置 | 训练脚本值 | 匹配 |
|-----------|-----------|-----------|------|
| hidden_size | 7168 | 7168 | ✓ |
| ffn_hidden_size | 18432 | 18432 | ✓ |
| num_attention_heads | 64 | 64 | ✓ |
| num_query_groups | 2 | 2 | ✓ |
| qk_head_dim / kv_channels | 128 | kv-channels=128 | ✓ |
| v_head_dim | 128 | kv-channels=128 | ✓ |
| num_experts | 128 | 128 | ✓ |
| moe_ffn_hidden_size | 12288 | 12288 | ✓ |
| first_k_dense_replace | 2 | 2 | ✓ |
| n_shared_experts | 1 | 1 | ✓ |
| moe_router_topk | 2 | 2 | ✓ |
| moe_router_num_groups | 8 | 8 | ✓ |
| moe_router_group_topk | 2 | 2 | ✓ |
| moe_router_topk_scaling_factor | 2.827 | 2.827 | ✓ |
| moe_router_enable_expert_bias | True | 启用 | ✓ |
| swiglu | True | 启用 | ✓ |
| untie_embeddings_and_output_weights | True | 启用 | ✓ |
| position_embedding_type | 'rope' | rope | ✓ |
| normalization | 'RMSNorm' | RMSNorm | ✓ |
| add_bias_linear | False | disable-bias-linear | ✓ |
| norm_epsilon | 1e-6 | 1e-6 | ✓ |
| bf16 | True | bf16 | ✓ |
| rotary_base | 50000.0 | 50000 | ✓ |
| vocab_size | 163840 | 163840 | ✓ |
| use_distributed_optimizer | True | 启用 | ✓ |
| qk_layernorm | 由参数控制 | 启用 | ✓ |
| use_mcore_models | True | 启用 | ✓ |

所有配置完全匹配。✓

#### 3.7.4 DualPipe/VPP 模式保存

VPP 模式下的保存格式：
```python
model_dict = {
    'checkpoint_version': 3.0,
    'iteration': 1,
    'args': ...,
}
for vpp_rank in range(self.vpp_size):
    model_key = f"model{vpp_rank}"
    model_dict[model_key] = mg_model[vpp_rank][ep_rank][tp_rank]
```

每个 checkpoint 文件包含 `model0` 和 `model1`（dualpipe 的 vpp_size=2）。这是 Megatron VPP 的标准保存格式。✓

Postprocess 放置逻辑：
```python
# DualPipe: norm + lm_head 放在 pp_rank=0, vpp_rank=last
if self.dualpipe and pp_rank == 0 and vpp_rank == self.vpp_size - 1:
    self.set_model_postprocess(...)

# 标准 PP: norm + lm_head 放在 pp_rank=last, vpp_rank=last
if not self.dualpipe and pp_rank == self.pp_size - 1 and vpp_rank == self.vpp_size - 1:
    self.set_model_postprocess(...)
```

与 DeepSeek3 一致。✓

---

## 4. 与 DeepSeek3 参考实现的详细对比

### 4.1 架构差异（设计性差异，非 Bug）

| 特性 | Kimi2 | DeepSeek3 | 说明 |
|------|-------|-----------|------|
| 注意力机制 | GQA (q/k/v_proj + o_proj) | MLA (q_a/kv_a + q_b/kv_b) | 根本架构差异 |
| QK LayerNorm | q_layernorm + k_layernorm | q_a_layernorm + kv_a_layernorm | GQA 独立 QK norm vs MLA 共享 KV norm |
| MTP 层 | 无 | 支持 | Kimi2 不需要 |
| MLA mm_split | 无 | 支持 | Kimi2 不需要 |
| expert_tp_size | 可配置 (默认 1) | 硬编码为 tp_size | Kimi2 更灵活 |
| Shared expert 切分位置 | expert 循环外 | expert 循环内 | Kimi2 效率更高 |
| bitsandbytes 导入 | 条件导入 | 顶层导入 | Kimi2 更友好 |
| NPU 硬编码 | 无 | .to('npu').cpu() | Kimi2 跨平台 |

### 4.2 代码质量改进

| 改进点 | Kimi2 | DeepSeek3 |
|--------|-------|-----------|
| 参数校验 | 范围校验 + 互斥校验 + 跨参校验 | 较简单 |
| Dense/MoE 检测 | 自动检测 + 警告 | 仅依赖索引 |
| Router bias | 可选 (None default) | 必选 |
| Router weight 截断 | 条件截断 | 无条件截断 |
| gc.collect() | 使用 | 未使用 |
| checkpoint args | 保存完整配置 | 未保存 |
| os.makedirs | exist_ok=True | 纯 PP 模式未设 exist_ok |

### 4.3 关键逻辑一致性验证

| 逻辑路径 | Kimi2 vs DeepSeek3 | 一致性 |
|---------|-------------------|--------|
| Dense MLP gate/up 交织 | 相同交织模式 | ✓ |
| Expert fc1 交织 | 相同 (配对 gate_i + up_i) | ✓ |
| Expert .t() 双重转置 | 相同 | ✓ |
| Grouped GEMM 3D view | 相同 reshape 链 | ✓ |
| EP 切分 (dim=0) | 相同 | ✓ |
| TP 切分 (dim=2 for fc1, dim=1 for fc2) | 相同 | ✓ |
| moe_tp_extend_ep 桶映射 | 相同 (ep*tp 线性映射) | ✓ |
| DualPipe 层映射 | 相同 (前后半分配) | ✓ |
| Checkpoint 目录命名 | 相同 (mp_rank_{tp:02}_{pp:03}_{ep:03}) | ✓ |
| VPP 保存格式 (model0, model1) | 相同 | ✓ |

---

## 5. 发现的问题与建议

### 5.1 已确认正确（无需修改）

| 项目 | 状态 |
|------|------|
| GQA QKV 拼接与 TP 切分 | ✓ 正确 |
| Dense MLP SwiGLU gate/up 交织 | ✓ 正确 |
| MoE expert 权重构建与 TP 切分 | ✓ 正确 |
| Shared expert 与独立 expert 的 TP 处理差异 | ✓ 正确 |
| DualPipe 层映射与权重放置 | ✓ 正确 |
| Pure PP / VPP / DualPipe 三模式支持 | ✓ 正确 |
| Noop 层处理 | ✓ 正确 |
| Checkpoint args 完整性 | ✓ 正确 |
| VPP 保存格式 (model0, model1, ...) | ✓ 正确 |
| 未消费权重检测 | ✓ 正确 |
| QK LayerNorm 可选支持 | ✓ 正确 |
| gc.collect() 内存管理 | ✓ 正确 |

### 5.2 建议改进（非 Bug，增强鲁棒性）

#### 5.2.1 训练配置 EP 与转换脚本默认值

Shell 脚本默认 `EP=8`，而训练脚本使用 `EP=64`。建议在 Shell 脚本注释中提醒用户：

```bash
# 注意: EP 默认值为 8，训练脚本使用 EP=64
# 转换时需保持与训练一致: EP=64 bash scripts/ckpt_convert_kimi2_hf2mcore.sh
```

#### 5.2.2 `num_query_groups` 与 `tp_size` 约束提醒

GQA 下 `num_query_groups=2`，这限制了 `tp_size` 最大为 2。虽然代码有校验，但建议在 Shell 脚本注释中明确说明。

---

## 6. 结论

### 6.1 总体评价

`convert_kimi2_hf2mcore.py` 实现质量较高，主要优点：

1. **架构适配正确**：GQA 注意力、SwiGLU MLP、MoE expert/shared expert 均正确实现
2. **并行支持完整**：TP/PP/EP/VPP/DualPipe 五种并行维度均支持
3. **防御性编程**：Dense/MoE 自动检测、shape 校验、未消费权重检查
4. **代码质量**：比 DeepSeek3 参考实现在多处有改进（条件导入、checkpoint args、shared expert 效率）
5. **与 DeepSeek3 兼容**：在共享的逻辑路径上（MoE expert 处理、checkpoint 格式、层映射），实现完全一致

### 6.2 未发现 Bug

经过逐行对比和 Shape 推导验证，**未发现功能性 Bug**。所有权重转换逻辑、TP/EP/PP 切分、保存格式均正确。

### 6.3 使用建议

```bash
# 推荐转换命令 (与训练配置一致)
TP=2 PP=8 EP=64 EXPERT_TP=1 \
SCHEDULES_METHOD=dualpipev \
bash scripts/ckpt_convert_kimi2_hf2mcore.sh
```
