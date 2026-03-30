# Kimi2-PCL 分布式推理部署指南 (vLLM & vLLM-Ascend)

Kimi2-PCL 是一个超大规模（1T级别）的 MoE 架构模型。要在生产环境或测试集群中进行推理，强烈推荐使用 **vLLM + Ray** 的多节点分布式架构，特别是针对昇腾 (Ascend NPU) 硬件进行了深度适配（即 `vllm-ascend`）。

以下是在集群中部署 Kimi2-PCL 推理服务的完整操作流程，包含必需的**源码级算子适配指南**与**集群部署流程**。

---

## 第一部分：vLLM 源码级适配指南 (核心必做)

由于 Kimi2-PCL 融合了 DeepSeek-V3 的 MoE 路由逻辑，但改用了 **GQA + QK LayerNorm**。虽然可以通过 `--trust-remote-code` 加载原生代码推理，但这会丢失 vLLM 的 **PagedAttention** 和 **FusedMoE** 核心加速特性，导致极易 OOM 且性能极差。

因此，**必须在 vLLM 源码内部实现原生的 `KimiK2ForCausalLM` 架构支持**。

### 1. 注册模型架构类型
在 vLLM 的模型注册表中注册 `kimi_k2` 模型类型，使其能够路由到我们新建的模型类。
**修改文件**：`vllm/model_executor/models/registry.py` 或 `vllm/model_executor/models/__init__.py`
```python
# 增加注册映射
_MODELS = {
    ...
    "kimi_k2": ("kimi_k2", "KimiK2ForCausalLM"),
    ...
}
```

### 2. 编写原生的模型实现文件
参考 vLLM 中的 `llama.py` 和 `deepseek_v2.py`，新建 `vllm/model_executor/models/kimi_k2.py`。需要重点实现以下组件：

#### A. Attention 层的定制 (GQA + QK LayerNorm)
Kimi2-PCL 在 Q 和 K 上应用了 LayerNorm，并且 Q/K/V 的 `head_dim` 不一致（Q/K 是 192，V 是 128）。
```python
class KimiK2Attention(nn.Module):
    def __init__(self, config, ...):
        # 1. 投影层：分别定义 Q, K, V 的并行线性层
        self.q_proj = ColumnParallelLinear(hidden_size, num_heads * q_head_dim, ...)
        self.k_proj = ColumnParallelLinear(hidden_size, num_kv_heads * q_head_dim, ...)
        self.v_proj = ColumnParallelLinear(hidden_size, num_kv_heads * v_head_dim, ...)
        
        # 2. QK LayerNorm (必须加在 RoPE 之前)
        self.q_layernorm = RMSNorm(q_head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = RMSNorm(q_head_dim, eps=config.rms_norm_eps)

        # 3. vLLM 原生 Attention 算子
        self.attn = Attention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position=config.max_position_embeddings,
            # 需确保 vLLM 算子支持不对称的 head_dim (Q/K=192, V=128)
        )

    def forward(self, hidden_states, ...):
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # 应用 QK LayerNorm
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)

        # 应用 RoPE
        q, k = self.rotary_emb(q, k, positions)

        # 传入 vLLM PagedAttention
        return self.attn(q, k, v, ...)
```

#### B. MoE 路由与融合计算定制
需要复用或微调 vLLM 的 `FusedMoE`，实现 Grouped Top-K 路由和共享专家。
```python
class KimiK2MoE(nn.Module):
    def __init__(self, config):
        # 1. 共享专家
        self.shared_experts = KimiK2MLP(config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)
        # 2. Router (带 group_topk 逻辑)
        self.router = KimiK2MoEGate(config)
        # 3. 路由专家：使用 vLLM FusedMoE 加速
        self.experts = FusedMoE(...)

    def forward(self, hidden_states):
        topk_weights, topk_ids = self.router(hidden_states)
        routed_out = self.experts(hidden_states, topk_weights, topk_ids)
        return routed_out + self.shared_experts(hidden_states)
```

### 3. FusedMoE / Router 权重加载映射
在 `load_weights` 函数中，需要将 HF 格式的权重拼装为 vLLM 算子格式：
- `q_proj`, `k_proj`, `v_proj` 直接映射。
- `gate_proj` 和 `up_proj` 按 `dim=0` 拼接并加载到 `FusedMoE` 的 `gate_up_proj` 中。
- `mlp.gate.weight` 映射到 `router.weight`。

### 4. Ascend (昇腾) NPU 算子特化适配
由于底层的 NPU 算子要求严格，需在 `vllm-ascend` 中确认以下两点：
1. **非对称 Head 维度支持**：检查 `vllm/attention/backends/ascend.py` 中的 FlashAttention/PagedAttention 是否支持 QK=192、V=128。若不支持，需在 Python 层对 V 进行 Zero-Padding（补齐到 192），在 Output 层截断。
2. **MoE Grouped GEMM 分布式**：Kimi2 是 EP=64。在 Ascend 上需确保 `vllm/model_executor/layers/fused_moe` 的昇腾实现能够处理 128 个专家的跨卡路由表。

---

## 第二部分：集群部署流程

当源码适配编译通过后，方可进行大规模集群部署。

### 2.1 基础设施与前置准备
- **节点规模**：如 16 个节点，每节点 8 张 NPU，共 128 NPU。
- **网络与存储**：节点间必须 SSH 免密登录，并挂载同一块共享存储（如 NFS / dtfs），确保模型权重、Docker 镜像压缩包在绝对路径上一致。
- **配置文件**：
  - `vllm-infer/node_list.txt`：填入所有集群节点的 IP 地址。
  - `vllm-infer/set_env.sh`：配置核心环境变量。

### 2.2 部署方案一：一键自动化部署（推荐）
在部署机执行：
```bash
bash vllm-infer/cluster_deploy_ray_vllm.sh
```
该脚本会在后台依次完成：
1. 分发镜像并拉起所有节点的 Docker 容器。
2. 在 Master 启动 Ray Head，在其余节点启动 Worker。
3. 在 Master 容器内执行 `vllm_model_server.sh` 拉起服务。

### 2.3 部署方案二：手动启动与调试
如果在排障阶段，可手动拉起：

1. **绑定高速网卡** (所有节点)：
   ```bash
   export GLOO_SOCKET_IFNAME=eth0
   export TP_SOCKET_IFNAME=eth0
   export HCCL_SOCKET_IFNAME=eth0
   ```
2. **启动 Ray 集群**：
   - Master: `ray start --head --port=6379 --node-ip-address=<master_ip>`
   - Worker: `ray start --address='<master_ip>:6379' --node-ip-address=<worker_ip>`
3. **启动 vLLM 服务** (在 Master 容器内)：
   ```bash
   vllm serve /path/to/Kimi2-PCL-Weights \
     --host 0.0.0.0 \
     --port 8000 \
     --served-model-name kimi2-pcl \
     --gpu-memory-utilization 0.9 \
     --max-model-len 16384 \
     --max-num-seqs 4096 \
     --max-num-batched-tokens 32768 \
     --enable-expert-parallel \
     --disable-prefix-caching \
     --trust-remote-code \
     --quantization ascend \
     --load-format safetensors \
     --dtype bfloat16 \
     --tensor-parallel-size 8 \
     --pipeline-parallel-size 16 \
     --distributed-executor-backend ray \
     --enforce-eager
   ```

**⚠️ 关键参数解析：**
- `--tensor-parallel-size 8`：张量并行 (TP)。**极度建议控制在单机 NPU 数量内**（单机 8 卡则设为 8），避免跨机 TP 导致通信瓶颈。
- `--pipeline-parallel-size 16`：流水线并行 (PP)。用于跨节点扩展。
- `--enable-expert-parallel`：**必须开启**。对于 MoE 模型极大降低跨节点通信开销。
- `--disable-prefix-caching`：**当前版本建议关闭**，以避免与 MoE/流水线并行的兼容性冲突。

---

## 第三部分：常见排错与调优 (Ascend 环境)

如果在 Ascend 环境下遇到算子不支持、精度报错或卡死，可在启动 vLLM 前注入以下环境变量进行规避：

```bash
# 允许自动类型转换 (FP32 -> FP16)
export ALLOW_FP32_TO_FP16=1
export TORCH_NPU_DTYPE_CONVERT_ENABLE=1
export ACLNN_ALLOW_FLOAT32_TO_FLOAT16=1

# 禁用可能有问题的部分融合算子 (如需 Fallback)
export ATB_GMM_OP_ENABLE=0
export ATB_MOE_ENABLE=0

# 若遇到严重卡死，可开启同步阻塞以捕获真实报错位置
export ASCEND_LAUNCH_BLOCKING=1
```

## 验证部署结果
当终端显示 `Uvicorn running on http://0.0.0.0:8000` 后，通过 OpenAI API 格式进行测试：
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi2-pcl",
    "messages": [{"role": "user", "content": "你好，请介绍一下你自己。"}],
    "max_tokens": 512,
    "stream": true
  }'
```