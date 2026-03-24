#!/usr/bin/env bash

# ==============================================================================
# vLLM Model Server Startup Script (Optimized for Kimi-K2-Base on 128 NPU Cluster)
# ==============================================================================
# 目标架构: 16节点 * 8 NPU (总计 128 NPU)
# 模型规格: DeepseekV3 架构 (MoE), hidden_size=7168, layers=61, routed_experts=384
# 优化方向: 最大化吞吐量、启用前缀缓存、分块预填充、多步调度等。

set -euo pipefail

# ------------------------------------------------------------------------------
# 1. 基础环境变量配置
# ------------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/llm_workspace_1P/robin/hfhub/models/moonshotai/Kimi-K2-Base}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2-base}"
HOST="${HOST:-}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# ------------------------------------------------------------------------------
# 2. 分布式并行配置 (核心调整)
# ------------------------------------------------------------------------------
# Kimi-K2 采用 MoE 架构，参数量巨大（总计约 600B+），在 64 NPU (8节点) 集群上，
# 我们采用张量并行(TP)结合流水线并行(PP)以及专家并行(EP)的组合。
# - TP (Tensor Parallel): 推荐节点内 TP=8，以利用高速的节点内互联 (HCCL)
# - PP (Pipeline Parallel): 根据当前 Ray 集群的实际资源量 (8个节点)，设为 8。
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-8}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-ray}"

# 专家并行 (Expert Parallel): 对于 DeepseekV3 架构极大地提升训练和推理效率
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
EXPERT_PARALLEL_SIZE="${EXPERT_PARALLEL_SIZE:-64}" # 默认让EP=TP*PP或让框架自动切分专家

# ------------------------------------------------------------------------------
# 3. 内存与量化配置
# ------------------------------------------------------------------------------
DTYPE="${DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-}" # 模型 config.json 已经指定了 quant_method: fp8
LOAD_FORMAT="${LOAD_FORMAT:-safetensors}"
# GPU(NPU) 内存利用率：保留一些显存给 KV Cache
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
# CPU 交换空间：以 GiB 为单位，处理 KV Cache 驱逐时的缓冲
SWAP_SPACE="${SWAP_SPACE:-64}"

# ------------------------------------------------------------------------------
# 4. 吞吐量与序列调度优化
# ------------------------------------------------------------------------------
# 允许的最大模型长度 (模型原生支持 131072, 但为了内存和吞吐折中，通常限制在 32k 或 64k)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# 最大的并发请求数量
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
# Chunked Prefill (分块预填充)：解耦 Prefill 和 Decode 阶段，提升并发
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
# Max Num Batched Tokens: 当启用 chunked-prefill 时，决定每个 step 处理的 token 数
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# ------------------------------------------------------------------------------
# 5. 高级加速特性与环境变量修正
# ------------------------------------------------------------------------------
# 启用 Prefix Caching: 对于多轮对话或大量重复 system prompt 极其有效
PREFIX_CACHING="${PREFIX_CACHING:-1}"
# Multi-step scheduling: 减少框架在各个 NPU 之间的调度通信开销
NUM_SCHEDULER_STEPS="${NUM_SCHEDULER_STEPS:-8}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}" # 如果支持 CUDA Graph/编译图，则设为0，否则设为1
AUTO_DETECT_FLAGS="${AUTO_DETECT_FLAGS:-1}"

# 在最新的 vLLM 中，可以通过这个变量关闭严格的节点 IP 绑定，允许更灵活的跨容器/多网卡 Actor 调度
export VLLM_RAY_PER_NODE_OBJECT_STORE_MEMORY=0

# ==============================================================================
# 环境预检与辅助函数
# ==============================================================================
if ! command -v vllm >/dev/null 2>&1; then
  echo "[ERROR] vllm not found in PATH" >&2
  exit 127
fi

if [[ ! -e "$MODEL_PATH" ]]; then
  echo "[ERROR] MODEL_PATH not found: $MODEL_PATH" >&2
  exit 2
fi

vllm_help() {
  vllm serve --help 2>/dev/null || true
}

choose_flag() {
  local help_text="$1"
  local preferred="$2"
  local fallback="$3"

  if [[ -n "$preferred" && "$help_text" == *"$preferred"* ]]; then
    echo "$preferred"
    return 0
  fi

  if [[ -n "$fallback" && "$help_text" == *"$fallback"* ]]; then
    echo "$fallback"
    return 0
  fi
  echo "$preferred"
}

# ==============================================================================
# 构建启动参数
# ==============================================================================
# 注意：由于我们在 start_ray_cluster.sh 中并没有传入自定义资源 "NPU"，
# 而是使用了内置资源 "num_gpus" 来适配 vLLM 的默认行为，
# 我们可以在启动 vllm 时告诉它我们的 Worker 就是普通的 GPU（实际上底层调的是 NPU）
# 否则 vLLM 可能强制要求自定义资源 "NPU"，从而找不到匹配的节点。

args=(
  serve
  "$MODEL_PATH"
  --trust-remote-code
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype "$DTYPE"
  --distributed-executor-backend "$DISTRIBUTED_EXECUTOR_BACKEND"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --swap-space "$SWAP_SPACE"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
)

# 动态探测的 Help 信息缓存
HELP_TEXT=""
if [[ "$AUTO_DETECT_FLAGS" == "1" ]]; then
    HELP_TEXT="$(vllm_help)"
fi

# ==============================================================================
# 动态特性检测与兼容性处理
# ==============================================================================

# 0. 尝试探测较新的参数 (--num-scheduler-steps)
if [[ "$AUTO_DETECT_FLAGS" == "1" ]]; then
  if [[ "$HELP_TEXT" == *"--num-scheduler-steps"* ]]; then
      args+=(--num-scheduler-steps "$NUM_SCHEDULER_STEPS")
  fi
else
  # 如果不自动探测，则为了兼容老版本先不加这个参数，或者你可以手动解开注释
  # args+=(--num-scheduler-steps "$NUM_SCHEDULER_STEPS")
  :
fi

# 日志级别参数检测 (老版本可能叫 --logging-level 或没有)
if [[ "$AUTO_DETECT_FLAGS" == "1" ]]; then
  if [[ "$HELP_TEXT" == *"--log-level"* ]]; then
      args+=(--log-level "$LOG_LEVEL")
  fi
fi

# 1. 量化与格式
if [[ -n "$QUANTIZATION" && "$QUANTIZATION" != "none" ]]; then
  args+=(--quantization "$QUANTIZATION")
fi

if [[ -n "$LOAD_FORMAT" ]]; then
  args+=(--load-format "$LOAD_FORMAT")
fi

# 2. Expert Parallel
if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]]; then
  ep_flag="--enable-expert-parallel"
  if [[ "$AUTO_DETECT_FLAGS" == "1" ]]; then
      ep_flag="$(choose_flag "$HELP_TEXT" "--enable-expert-parallel" "--enable_expert_parallel")"
  fi
  args+=("$ep_flag")
fi

# 3. Chunked Prefill
if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
  args+=(--enable-chunked-prefill)
fi

# 4. Prefix Caching
if [[ "$PREFIX_CACHING" == "1" ]]; then
  pc_flag="--enable-prefix-caching"
  if [[ "$AUTO_DETECT_FLAGS" == "1" ]]; then
      pc_flag="$(choose_flag "$HELP_TEXT" "--enable-prefix-caching" "--enable_prefix_caching")"
  fi
  args+=("$pc_flag")
else
  dpc_flag="--disable-prefix-caching"
  if [[ "$AUTO_DETECT_FLAGS" == "1" ]]; then
      dpc_flag="$(choose_flag "$HELP_TEXT" "--disable-prefix-caching" "--disable_prefix_caching")"
  fi
  args+=("$dpc_flag")
fi

# 5. Enforce Eager
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  args+=(--enforce-eager)
fi

if [[ -n "$HOST" ]]; then
  args+=(--host "$HOST")
fi

if [[ -n "$PORT" ]]; then
  args+=(--port "$PORT")
fi

# ==============================================================================
# 启动与监控包装 (自动重试与健康检查)
# ==============================================================================
echo "=========================================================="
echo "[INFO] Starting vLLM Server with optimized configuration:"
echo "[INFO] TP: $TENSOR_PARALLEL_SIZE | PP: $PIPELINE_PARALLEL_SIZE"
echo "[INFO] Max Model Len: $MAX_MODEL_LEN | Chunked Prefill: $ENABLE_CHUNKED_PREFILL"
echo "[INFO] Prefix Caching: $PREFIX_CACHING | Multi-step: $NUM_SCHEDULER_STEPS"
echo "[INFO] Executing command: vllm ${args[*]} $@"
echo "=========================================================="

MAX_RETRIES=3
RETRY_COUNT=0
RETRY_DELAY=10

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # 使用 exec 替换当前 shell，但是为了外层能做健康检查或重启，我们放在后台运行或通过 wait 捕获
    # 在生产脚本中，最好直接由 systemd 或 supervisord 接管。
    # 这里我们采用阻塞运行，如果退出码非 0 则尝试重启。
    set +e
    vllm "${args[@]}" "$@"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[INFO] vLLM server exited normally."
        break
    else
        echo "[ERROR] vLLM server crashed with exit code $EXIT_CODE."
        RETRY_COUNT=$((RETRY_COUNT+1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "[INFO] Retrying in $RETRY_DELAY seconds... (Attempt $RETRY_COUNT of $MAX_RETRIES)"
            sleep $RETRY_DELAY
        else
            echo "[FATAL] Maximum retries reached. Exiting."
            exit $EXIT_CODE
        fi
    fi
done
