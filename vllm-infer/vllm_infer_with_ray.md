# Ray 分布式推理（vLLM Ascend）

本文档给出一套在 Ascend NPU 环境下，使用 vLLM + Ray 进行多节点分布式推理的推荐流程，并将实践笔记整理为可直接执行的命令与脚本化方式。

## 目标与假设

- 目标：在多节点（示例：16 节点 × 8 NPU）上启动 vLLM 在线推理服务（OpenAI 兼容 API）。
- 假设：节点间网络可互通；每个节点可访问共享存储（用于镜像 tar、模型权重与缓存）。
- 推荐：尽量使用 host 网络（`--net=host`）简化 Ray 通信与端口配置。

## 1. 通信与硬件检查

### 1.1 物理层要求

- 物理机位于同一局域网，具备网络连通性。
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

如需挂载分布式存储（示例为 dtfs）：

```bash
mount -t dtfs  /llm_workspace_1P  /llm_workspace_1P
```

建议将以下内容放在所有节点可见的共享目录下：

- 镜像包：`/llm_workspace_1P/robin/hfhub/docker/image/vllm-ascend.main-a3.tar`
- 模型权重：例如 `/llm_workspace_1P/robin/hfhub/models/moonshotai/Kimi-K2-Base`
- 缓存目录：`/root/.cache`（或自定义路径），确保多节点一致（避免重复下载/编译）

## 3. 容器与镜像（推荐脚本化）

### 3.1 镜像存在性与加载

镜像标签：

- `quay.io/ascend/vllm-ascend:main-a3`

如果节点上不存在该镜像，需要从共享存储加载：

```bash
docker load -i /llm_workspace_1P/robin/hfhub/docker/image/vllm-ascend.main-a3.tar
```

### 3.2 启动容器

推荐使用脚本启动容器（每节点执行）：

- [ascend_infer_docker_run.sh](file:///Users/jianzhengnie/work_dir/Kimi2-PCL/vllm-infer/ascend_infer_docker_run.sh)

该脚本默认容器名为 `vllm-ascend-env-a3`，并挂载共享目录 `/llm_workspace_1P`、缓存目录 `/root/.cache` 等。

如需在容器内使用 SSH（例如访问私有仓库/私有存储），可将宿主机 `/root/.ssh` 挂载进容器（脚本中已支持/可自行添加）。

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

## 4. Ray 集群启动

### 4.1 推荐脚本（一键准备 + Ray + vLLM）

如果你已在控制机上配置好可免密 SSH 到所有节点，可使用：

- [cluster_deploy_ray_vllm.sh](file:///Users/jianzhengnie/work_dir/Kimi2-PCL/vllm-infer/cluster_deploy_ray_vllm.sh)
- [nodel_liist.txt](file:///Users/jianzhengnie/work_dir/Kimi2-PCL/vllm-infer/nodel_liist.txt)

常见用法：

```bash
bash vllm-infer/cluster_deploy_ray_vllm.sh
```

仅执行其中某一步：

```bash
bash vllm-infer/cluster_deploy_ray_vllm.sh --prepare-only
bash vllm-infer/cluster_deploy_ray_vllm.sh --ray-only
bash vllm-infer/cluster_deploy_ray_vllm.sh --serve-only
```

重要：脚本默认会在容器内执行 `VLLM_START_SCRIPT` 指定的启动脚本路径。若你的容器未挂载仓库目录，请将 `vllm_start.sh` 放到容器可见路径（例如共享目录 `/llm_workspace_1P/...`），并通过环境变量覆盖：

```bash
VLLM_START_SCRIPT=/llm_workspace_1P/robin/hfhub/scripts/vllm_start.sh bash vllm-infer/cluster_deploy_ray_vllm.sh --serve-only
```

### 4.2 手动启动（容器内执行）

主节点（Head）：

```bash
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray stop -f || true
ray start --head --port=6379 --node-ip-address={local_ip}
```

从节点（Worker）：

```bash
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray stop -f || true
ray start --address='{head_node_ip}:6379' --node-ip-address={local_ip}
```

验证：

```bash
ray status
ray list nodes
```

调试/规避部分通信问题时，可按需在 `ray start` 前设置（不保证所有环境都需要）：

```bash
export HCCL_P2P_DISABLE=1
export ACLNN_ALLOW_DTYPE_CONVERT=1
```

## 5. 启动 vLLM 分布式推理服务

### 5.1 并行策略建议（TP/PP）

- `--tensor-parallel-size`（TP）：尽量不要超过单机 NPU 数（通常 8）。跨机器做 TP 会带来显著通信开销，除非网络带宽/拓扑非常强。
- `--pipeline-parallel-size`（PP）：跨节点扩展时更常用。对于 16 节点 × 8 NPU 场景，常见配置是 TP=8、PP=16（每节点 8 卡做 TP，16 个节点做流水线切分）。
- 注意：某些量化/加载格式可能对 PP 有限制。如遇到 PP 相关报错，优先在同一版本镜像下用较小 PP 验证，再逐步扩大规模。

### 5.2 推荐启动命令（示例）

示例模型路径与服务名请替换为你的实际值。

2 节点 × 8 NPU（共 16 NPU）示例：

```bash
vllm serve /llm_workspace_1P/robin/hfhub/models/moonshotai/Kimi-K2-Base \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name kimi-k2-base \
  --gpu-memory-utilization 0.9 \
  --enable_expert_parallel \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --quantization ascend \
  --load-format safetensors \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --enforce-eager \
  --distributed-executor-backend ray
```

16 节点 × 8 NPU（共 128 NPU）建议从 TP=8、PP=16 开始：

```bash
vllm serve /llm_workspace_1P/robin/hfhub/models/moonshotai/Kimi-K2-Base \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name kimi-k2-base \
  --gpu-memory-utilization 0.9 \
  --enable_expert_parallel \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --quantization ascend \
  --load-format safetensors \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 16 \
  --enforce-eager \
  --distributed-executor-backend ray
```

### 5.3 常见稳定性开关（按需）

当遇到类型转换/算子不支持等问题时，可尝试以下环境变量（会影响性能或精度，建议用于定位问题）：

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

如需强行规避多数类型转换报错，可临时使用 `--dtype float32` 验证可行性（显存开销更大）：

```bash
vllm serve /mnt/model_test/models/vllm-ascend/Kimi-K2-Instruct-W8A8 \
  --served-model-name kimi-k2-thinking \
  --gpu-memory-utilization 0.7 \
  --enable_expert_parallel \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --quantization ascend \
  --load-format safetensors \
  --dtype float32 \
  --tensor-parallel-size 64 \
  --enforce-eager \
  --distributed-executor-backend ray
```

### 5.4 并发与长度配置（按需）

在高并发/长上下文场景中，可逐步增大以下参数（注意显存与吞吐的权衡）：

```bash
vllm serve /llm_workspace_1P/robin/hfhub/models/moonshotai/Kimi-K2-Base \
  --served-model-name kimi-k2-base \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 4096 \
  --max-num-batched-tokens 32768 \
  --max-model-len 16384 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 16 \
  --distributed-executor-backend ray
```

定位卡住/算子报错时，可按需打开阻塞调试（会显著影响性能）：

```bash
export ASCEND_LAUNCH_BLOCKING=1
```

## 6. API 测试

Chat Completions（推荐）：

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

如模型支持思考/推理内容输出，可加 `include_thought`（取决于模型与服务实现）：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2-thinking",
    "messages": [{"role": "user", "content": "我想设计一个数据处理流程，请给出参考"}],
    "max_tokens": 200,
    "stream": false,
    "include_thought": true
  }'
```

Completions（兼容接口）：

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

## 7. 常见排错要点

- 环境变量必须在 `ray start` 之前设置；变更后需 `ray stop -f` 再重新 `ray start`
- 网络接口选择：`GLOO_SOCKET_IFNAME`、`TP_SOCKET_IFNAME` 应与实际可通信网卡匹配
- 性能与并行：优先将 TP 控制在单机范围内，用 PP 扩展到更多节点；不要直接上来就跨机 TP=128
- 模型配置：部分模型可能需要调整 `config.json`（例如 `torch_dtype`）以匹配算子与精度支持
