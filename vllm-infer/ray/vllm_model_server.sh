#!/usr/bin/env bash

# ==============================================================================
# vLLM Model Server Startup Script (Optimized for Kimi-K2-Base on NPU Cluster)
# ==============================================================================
# 目标架构: 多节点 NPU 集群 (建议 8+ 节点, 每节点 8 NPU)
# 模型规格: DeepseekV3 架构 (MoE)
#   - 参数量: ~32B (激活) / ~600B+ (总计)
#   - Hidden Size: 7168
#   - 层数: 61
#   - 路由专家: 384 (每 token 激活 8 个)
#   - 最大长度: 131072
#   - 量化: FP8 (dynamic, e4m3)
# 优化方向: 最大化吞吐量、启用前缀缓存、分块预填充、多步调度
#
# 使用方法:
#   1. 默认启动: ./vllm_model_server.sh
#   2. 指定配置文件: VLLM_ENV_FILE=/path/to/env.sh ./vllm_model_server.sh
#   3. 环境变量覆盖: MODEL_PATH=/path/to/model ./vllm_model_server.sh
#
# 配置文件:
#   - 默认加载当前目录下的 vllm_server_env.sh
#   - 可通过 VLLM_ENV_FILE 环境变量指定其他配置文件
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# 加载环境变量配置文件
# ------------------------------------------------------------------------------

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认配置文件路径
DEFAULT_ENV_FILE="$SCRIPT_DIR/vllm_server_env.sh"

# 可通过环境变量指定其他配置文件
VLLM_ENV_FILE="${VLLM_ENV_FILE:-$DEFAULT_ENV_FILE}"

# 加载配置文件
if [[ -f "$VLLM_ENV_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$VLLM_ENV_FILE"
    echo "[INFO] 已加载配置文件: $VLLM_ENV_FILE"
elif [[ -f "$DEFAULT_ENV_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$DEFAULT_ENV_FILE"
    echo "[INFO] 已加载默认配置文件: $DEFAULT_ENV_FILE"
else
    echo "[WARN] 未找到配置文件，使用内置默认值"
    echo "[INFO] 可通过创建 $DEFAULT_ENV_FILE 或使用 VLLM_ENV_FILE 环境变量指定配置文件"
fi

# ------------------------------------------------------------------------------
# 设置默认值 (如果配置文件未设置)
# ------------------------------------------------------------------------------

# 1. 基础配置
MODEL_PATH="${MODEL_PATH:-$HOME/hfhub/models/moonshotai/Kimi-K2-Base}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2-base}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# 2. 分布式并行配置
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-8}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-ray}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
EXPERT_PARALLEL_SIZE="${EXPERT_PARALLEL_SIZE:-$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))}"

# 3. 内存与量化配置
DTYPE="${DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-fp8}"
LOAD_FORMAT="${LOAD_FORMAT:-safetensors}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SWAP_SPACE="${SWAP_SPACE:-128}"

# 4. 吞吐量与序列调度优化
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_TOKENS_PER_SEQUENCE="${MAX_TOKENS_PER_SEQUENCE:-32768}"

# 5. 高级加速特性
PREFIX_CACHING="${PREFIX_CACHING:-1}"
NUM_SCHEDULER_STEPS="${NUM_SCHEDULER_STEPS:-8}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
MAX_SEQ_LEN_TO_CAPTURE="${MAX_SEQ_LEN_TO_CAPTURE:-8192}"
AUTO_DETECT_FLAGS="${AUTO_DETECT_FLAGS:-1}"

# 6. vLLM 环境变量 (设置 export)
export VLLM_RAY_PER_NODE_OBJECT_STORE_MEMORY="${VLLM_RAY_PER_NODE_OBJECT_STORE_MEMORY:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_MOE_PADDING="${VLLM_MOE_PADDING:-1}"
export VLLM_SKIP_CUDA_VERSION_CHECK="${VLLM_SKIP_CUDA_VERSION_CHECK:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-$LOG_LEVEL}"

# 7. API 和监控配置
API_KEY="${API_KEY:-}"
ENABLE_METRICS="${ENABLE_METRICS:-1}"
METRICS_PORT="${METRICS_PORT:-8001}"
DISABLE_LOG_REQUESTS="${DISABLE_LOG_REQUESTS:-0}"
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-*}"

# 8. 启动与重试配置
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-10}"

# ------------------------------------------------------------------------------
# 环境预检与辅助函数
# ==============================================================================

# 检查 vLLM 是否安装
if ! command -v vllm >/dev/null 2>&1; then
    echo "[ERROR] vllm not found in PATH" >&2
    echo "[INFO] 请安装 vLLM: pip install vllm" >&2
    exit 127
fi

# 检查模型路径
if [[ ! -e "$MODEL_PATH" ]]; then
    echo "[ERROR] MODEL_PATH not found: $MODEL_PATH" >&2
    echo "[INFO] 请设置正确的 MODEL_PATH 环境变量或修改配置文件" >&2
    exit 2
fi

# 检查关键模型文件
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[ERROR] config.json not found in $MODEL_PATH" >&2
    exit 2
fi

# 获取 vLLM 帮助信息
vllm_help() {
    vllm serve --help 2>/dev/null || true
}

# 选择支持的参数 (兼容不同版本)
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

# 检查参数是否支持
has_flag() {
    local help_text="$1"
    local flag="$2"
    [[ "$help_text" == *"$flag"* ]]
}

# ------------------------------------------------------------------------------
# 构建启动参数
# ==============================================================================
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

# ------------------------------------------------------------------------------
# 动态特性检测与兼容性处理
# ==============================================================================

# 1. 量化配置
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

# 5. Multi-step Scheduling
if [[ "$AUTO_DETECT_FLAGS" == "1" ]]; then
    if has_flag "$HELP_TEXT" "--num-scheduler-steps"; then
        args+=(--num-scheduler-steps "$NUM_SCHEDULER_STEPS")
    fi
else
    args+=(--num-scheduler-steps "$NUM_SCHEDULER_STEPS")
fi

# 6. Enforce Eager / CUDA Graph
if [[ "$ENFORCE_EAGER" == "1" ]]; then
    args+=(--enforce-eager)
else
    if [[ "$AUTO_DETECT_FLAGS" == "1" ]]; then
        if has_flag "$HELP_TEXT" "--max-seq-len-to-capture"; then
            args+=(--max-seq-len-to-capture "$MAX_SEQ_LEN_TO_CAPTURE")
        fi
    else
        args+=(--max-seq-len-to-capture "$MAX_SEQ_LEN_TO_CAPTURE")
    fi
fi

# 7. 日志级别
if [[ "$AUTO_DETECT_FLAGS" == "1" ]]; then
    if has_flag "$HELP_TEXT" "--log-level"; then
        args+=(--log-level "$LOG_LEVEL")
    elif has_flag "$HELP_TEXT" "--logging-level"; then
        args+=(--logging-level "$LOG_LEVEL")
    fi
fi

# 8. API 配置
if [[ -n "$HOST" ]]; then
    args+=(--host "$HOST")
fi

if [[ -n "$PORT" ]]; then
    args+=(--port "$PORT")
fi

if [[ -n "$API_KEY" ]]; then
    args+=(--api-key "$API_KEY")
fi

if [[ -n "$ALLOWED_ORIGINS" ]]; then
    if [[ "$AUTO_DETECT_FLAGS" == "1" ]]; then
        if has_flag "$HELP_TEXT" "--allowed-origins"; then
            args+=(--allowed-origins "$ALLOWED_ORIGINS")
        fi
    else
        args+=(--allowed-origins "$ALLOWED_ORIGINS")
    fi
fi

# 9. 监控配置
if [[ "$ENABLE_METRICS" == "1" ]]; then
    if [[ "$AUTO_DETECT_FLAGS" == "1" ]]; then
        if has_flag "$HELP_TEXT" "--enable-metrics"; then
            args+=(--enable-metrics)
            if has_flag "$HELP_TEXT" "--metrics-port"; then
                args+=(--metrics-port "$METRICS_PORT")
            fi
        fi
    else
        args+=(--enable-metrics)
        args+=(--metrics-port "$METRICS_PORT")
    fi
fi

if [[ "$DISABLE_LOG_REQUESTS" == "1" ]]; then
    if [[ "$AUTO_DETECT_FLAGS" == "1" ]]; then
        if has_flag "$HELP_TEXT" "--disable-log-requests"; then
            args+=(--disable-log-requests)
        fi
    else
        args+=(--disable-log-requests)
    fi
fi

# ------------------------------------------------------------------------------
# 打印配置摘要
# ==============================================================================
echo "================================================================================"
echo "[INFO] vLLM Server 配置摘要:"
echo "================================================================================"
echo "  模型路径:           $MODEL_PATH"
echo "  服务名称:           $SERVED_MODEL_NAME"
echo "  监听地址:           $HOST:$PORT"
echo "--------------------------------------------------------------------------------"
echo "  并行配置:"
echo "    - 张量并行 (TP):  $TENSOR_PARALLEL_SIZE"
echo "    - 流水线并行 (PP): $PIPELINE_PARALLEL_SIZE"
echo "    - 专家并行 (EP):   $EXPERT_PARALLEL_SIZE (启用: $ENABLE_EXPERT_PARALLEL)"
echo "    - 分布式后端:      $DISTRIBUTED_EXECUTOR_BACKEND"
echo "--------------------------------------------------------------------------------"
echo "  内存与量化:"
echo "    - 数据类型:        $DTYPE"
echo "    - 量化方式:        $QUANTIZATION"
echo "    - GPU 内存利用率:  $GPU_MEMORY_UTILIZATION"
echo "    - CPU 交换空间:    ${SWAP_SPACE}GiB"
echo "--------------------------------------------------------------------------------"
echo "  序列调度:"
echo "    - 最大模型长度:    $MAX_MODEL_LEN"
echo "    - 最大并发序列:    $MAX_NUM_SEQS"
echo "    - 最大批处理:      $MAX_NUM_BATCHED_TOKENS"
echo "--------------------------------------------------------------------------------"
echo "  加速特性:"
echo "    - 分块预填充:      $ENABLE_CHUNKED_PREFILL"
echo "    - 前缀缓存:        $PREFIX_CACHING"
echo "    - 多步调度:        $NUM_SCHEDULER_STEPS"
echo "    - Eager 模式:      $ENFORCE_EAGER"
echo "--------------------------------------------------------------------------------"
echo "  环境变量:"
echo "    - 注意力后端:      ${VLLM_ATTENTION_BACKEND:-默认}"
echo "    - Worker 方法:     ${VLLM_WORKER_MULTIPROC_METHOD:-默认}"
echo "    - MoE 填充:        ${VLLM_MOE_PADDING:-默认}"
echo "    - 日志级别:        $VLLM_LOGGING_LEVEL"
echo "--------------------------------------------------------------------------------"
echo "  监控配置:"
echo "    - 指标导出:        $ENABLE_METRICS (端口: $METRICS_PORT)"
echo "    - API 认证:        $([ -n "$API_KEY" ] && echo "已启用" || echo "未启用")"
echo "================================================================================"
echo "[INFO] 启动命令: vllm ${args[*]}"
echo "================================================================================"

# ------------------------------------------------------------------------------
# 启动与监控包装 (自动重试与健康检查)
# ==============================================================================
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    set +e
    vllm "${args[@]}" "$@"
    EXIT_CODE=$?
    set -e
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[INFO] vLLM server exited normally."
        break
    elif [ $EXIT_CODE -eq 130 ] || [ $EXIT_CODE -eq 137 ]; then
        # 130 = Ctrl+C, 137 = SIGKILL (通常由用户或系统发起)
        echo "[INFO] vLLM server terminated by signal (exit code $EXIT_CODE)."
        exit 0
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
