# `Ray` 分布式推理（vLLM Ascend）

本文档面向 Ascend NPU 环境，整理一套使用 vLLM + Ray 进行多节点分布式推理的推荐流程。内容以“可直接执行”为目标，给出脚本化与手动两种方式，并提供常用模型的参数模板与排障要点。

## 快速开始（推荐）

已在仓库中提供一键脚本，覆盖镜像检查/加载、容器启动、Ray 主从加入与 vLLM 服务启动：

- [cluster_deploy_ray_vllm.sh](./vllm-infer/cluster_deploy_ray_vllm.sh)
- [node_list.txt](./vllm-infer/node_list.txt)
- [ascend_infer_docker_run.sh](./vllm-infer/ascend_infer_docker_run.sh)

```bash
bash vllm-infer/cluster_deploy_ray_vllm.sh
```

仅执行某一步：

```bash
bash vllm-infer/cluster_deploy_ray_vllm.sh --prepare-only
bash vllm-infer/cluster_deploy_ray_vllm.sh --ray-only
bash vllm-infer/cluster_deploy_ray_vllm.sh --serve-only
```

脚本会在容器内执行 `VLLM_START_SCRIPT` 指定的启动脚本。默认使用仓库内的：

- `./vllm-infer/vllm_model_server.sh`

如果你的容器没有挂载整个仓库目录，请把启动脚本放到容器可见的共享路径，并通过环境变量覆盖，例如：

```bash
VLLM_START_SCRIPT=/llm_workspace_1P/robin/hfhub/scripts/vllm_model_server.sh \
  bash vllm-infer/cluster_deploy_ray_vllm.sh --serve-only
```

## 0. 目标与前置条件

- 目标：在多节点（示例：16 节点 × 8 NPU）上启动 vLLM 在线推理服务（OpenAI 兼容 API）。
- 前置：节点间网络可互通；各节点可访问共享存储（镜像 tar、模型权重与缓存）；建议使用 host 网络（`--net=host`）。

## 1. 通信与硬件检查

### 1.1 物理层要求

- 节点位于同一局域网，具备网络连通性。
- NPU 互联链路正常（光模块/交换机端口状态正常）。

### 1.2 NPU 网络与互连验证（每节点执行）

```bash
for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done
for i in {0..7}; do hccn_tool -i $i -link -g ; done
for i in {0..7}; do hccn_tool -i $i -net_health -g ; done
for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done
for i in {0..7}; do hccn_tool -i $i -gateway -g ; done
cat /etc/hccn.conf
```

获取 NPU IP 并做跨节点 ping（示例替换为实际 IP）：

```bash
for i in {0..7}; do hccn_tool -i $i -ip -g | grep ipaddr; done
hccn_tool -i 0 -ping -g address 10.20.0.20
```

## 2. 共享存储与模型准备

多节点推理要求：模型权重路径与缓存路径在所有节点上保持一致，并且容器内可见。

如需挂载分布式存储（示例为 dtfs，按实际环境替换）：

```bash
mount -t dtfs  /llm_workspace_1P  /llm_workspace_1P
```

建议统一使用的共享路径示例：

- 镜像包：`/llm_workspace_1P/robin/hfhub/docker/image/vllm-ascend.main-a3.tar`
- 模型权重：`/llm_workspace_1P/robin/hfhub/models/...`
- 缓存目录：`/root/.cache`（或自定义路径）

## 3. 容器与镜像

### 3.1 镜像标签与加载

镜像标签：

- `quay.io/ascend/vllm-ascend:main-a3`

如果节点上不存在该镜像，可从共享存储加载：

```bash
docker load -i /llm_workspace_1P/robin/hfhub/docker/image/vllm-ascend.main-a3.tar
```

### 3.2 启动容器（推荐脚本化）

推荐使用脚本启动容器（每节点执行）：

- [ascend_infer_docker_run.sh](./vllm-infer/ascend_infer_docker_run.sh)

该脚本默认容器名为 `vllm-ascend-env-a3`，并挂载共享目录 `/llm_workspace_1P`、缓存目录 `/root/.cache` 等。

部分场景需要在启动容器时增加 HCCL 缓冲相关环境变量，可在 `docker run` 中追加：

```bash
-e HCCL_BUFFSIZE=1024 \
-e HCCL_BUFFER_FILE_SIZE=1024 \
```

### 3.3 容器内环境初始化（按需）

部分镜像/环境需要手动 source Ascend 运行环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## 4. Ray 集群启动（手动方式）

### 4.1 关键环境变量

Ray 启动前建议设置（按实际网卡/网络环境调整）：

```bash
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLOO_SOCKET_IFNAME=<nic_name>
export TP_SOCKET_IFNAME=<nic_name>
export HCCL_SOCKET_IFNAME=<nic_name>
```

调试/规避部分通信问题时，可按需在 `ray start` 前设置：

```bash
export HCCL_P2P_DISABLE=1
export ACLNN_ALLOW_DTYPE_CONVERT=1
```

### 4.2 启动命令

主节点（Head）：

```bash
ray stop -f || true
ray start --head --port=6379 --node-ip-address=<head_ip>
```

从节点（Worker）：

```bash
ray stop -f || true
ray start --address='<head_ip>:6379' --node-ip-address=<worker_ip>
```

验证：

```bash
ray status
ray list nodes
```

## 5. 启动 vLLM 推理服务

### 5.1 并行策略（TP/PP）

- TP（`--tensor-parallel-size`）：优先控制在单机 NPU 数以内（通常 8），跨机 TP 会引入较大的通信开销。
- PP（`--pipeline-parallel-size`）：跨节点扩展时更常用。示例：16 节点 × 8 NPU 可从 TP=8、PP=16 起步。

### 5.2 典型启动模板

若你使用的是仓库脚本，优先通过脚本参数/环境变量管理；若直接手动跑 `vllm serve`，建议显式指定：

- `--distributed-executor-backend ray`
- `--host 0.0.0.0`（需要对外提供服务时）
- `--served-model-name <name>`（API 请求中 `model` 使用该名字）

### 5.3 示例：Qwen3-30B

> **配置建议**：Qwen3-30B 是 Dense 模型，推荐开启 `--enable-prefix-caching` 以大幅提升多轮对话与长上下文的吞吐性能；同时加入 `--served-model-name qwen3-30b` 方便 API 路由。在 NPU 显存允许的情况下，可将 `max-model-len` 扩展至 32768。

单节点（8 NPU）：

```bash
vllm serve /llm_workspace_1P/robin/hfhub/models/Qwen/Qwen3-30B-A3B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3-30b \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enforce-eager
```

两节点（2 × 8 NPU）：

```bash
vllm serve /llm_workspace_1P/robin/hfhub/models/Qwen/Qwen3-30B-A3B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3-30b \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --enforce-eager
```

### 5.4 示例：Kimi-K2-Base

> **配置建议**：Kimi-K2 属于大型 MoE（混合专家）架构，必须开启 `--enable-expert-parallel` 以减少跨节点通信开销；同时当前版本下建议关闭 Prefix Caching（`--disable-prefix-caching`），避免与 MoE/PP 产生兼容性问题。可通过增大 `max-num-seqs` 和 `max-num-batched-tokens` 提升系统并发上限。

两节点（2 × 8 NPU，16 NPU）：

```bash
vllm serve /llm_workspace_1P/robin/hfhub/models/moonshotai/Kimi-K2-Base \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name kimi-k2-base \
  --gpu-memory-utilization 0.9 \
  --max-model-len 16384 \
  --max-num-seqs 4096 \
  --max-num-batched-tokens 32768 \
  --enable-expert-parallel \
  --trust-remote-code \
  --disable-prefix-caching \
  --quantization ascend \
  --load-format safetensors \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --enforce-eager
```

16 节点（16 × 8 NPU，128 NPU）从 TP=8、PP=16 起步：

```bash
vllm serve /llm_workspace_1P/robin/hfhub/models/moonshotai/Kimi-K2-Base \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name kimi-k2-base \
  --gpu-memory-utilization 0.9 \
  --max-model-len 16384 \
  --max-num-seqs 4096 \
  --max-num-batched-tokens 32768 \
  --enable-expert-parallel \
  --trust-remote-code \
  --disable-prefix-caching \
  --quantization ascend \
  --load-format safetensors \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 16 \
  --distributed-executor-backend ray \
  --enforce-eager
```

## 6. 性能与稳定性配置（按需）

### 6.1 常用环境变量

当遇到类型转换/算子不支持等问题时，可按需尝试：

```bash
export ALLOW_FP32_TO_FP16=1
export ATB_OPERATION_AUTOTUNE=1
export ATB_CONV_TYPE_FLOAT16=1
export ATB_GMM_OP_ENABLE=0
export ATB_MOE_ENABLE=0
export TORCH_NPU_DTYPE_CONVERT_ENABLE=1
export ACLNN_ALLOW_FLOAT32_TO_FLOAT16=1
export MOE_GMM_ALIGN_DTYPE=1
```

定位卡住/算子报错时，可按需打开阻塞调试：

```bash
export ASCEND_LAUNCH_BLOCKING=1
```

### 6.2 并发与长度参数

如果硬件显存足够或模型需要更高的并发处理能力，可通过以下参数调整系统的最大调度序列数：

```bash
--max-num-seqs 4096 \
--max-num-batched-tokens 32768 \
--max-model-len 16384
```

### 6.3 float32 兜底验证

如需快速判断是否为精度/类型问题，可临时使用 `--dtype float32` 验证（显存开销更大）：

```bash
vllm serve /mnt/model_test/models/vllm-ascend/Kimi-K2-Instruct-W8A8 \
  --served-model-name kimi-k2-thinking \
  --gpu-memory-utilization 0.7 \
  --enable-expert-parallel \
  --trust-remote-code \
  --disable-prefix-caching \
  --quantization ascend \
  --load-format safetensors \
  --dtype float32 \
  --tensor-parallel-size 64 \
  --distributed-executor-backend ray \
  --enforce-eager
```

## 7. API 测试

Chat Completions：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2-base",
    "messages": [{"role": "user", "content": "你是什么模型？"}],
    "max_tokens": 200,
    "stream": false
  }'
```

Completions：

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2-base",
    "prompt": "tell me how to sleep well",
    "max_completion_tokens": 100,
    "temperature": 0
  }'
```

## 8. 常见排错要点

- 变量生效顺序：环境变量必须在 `ray start` 前设置；变更后需 `ray stop -f` 再重新 `ray start`
- 网卡选择：`GLOO_SOCKET_IFNAME`、`TP_SOCKET_IFNAME`、`HCCL_SOCKET_IFNAME` 要与实际可通信网卡匹配
- 路径一致性：模型路径与缓存路径需要在所有节点一致，并且容器内可见
- 并行策略：优先将 TP 控制在单机范围内，用 PP 扩展到更多节点
