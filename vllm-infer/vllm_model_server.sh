#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/llm_workspace_1P/robin/hfhub/models/moonshotai/Kimi-K2-Base}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2-base}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
QUANTIZATION="${QUANTIZATION:-ascend}"
LOAD_FORMAT="${LOAD_FORMAT:-safetensors}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-64}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
EXPERT_PARALLEL_FLAG="${EXPERT_PARALLEL_FLAG:-}"

PREFIX_CACHING="${PREFIX_CACHING:-0}"
ENABLE_PREFIX_CACHING_FLAG="${ENABLE_PREFIX_CACHING_FLAG:---enable-prefix-caching}"
DISABLE_PREFIX_CACHING_FLAG="${DISABLE_PREFIX_CACHING_FLAG:---disable-prefix-caching}"

DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-ray}"

HOST="${HOST:-}"
PORT="${PORT:-}"

AUTO_DETECT_FLAGS="${AUTO_DETECT_FLAGS:-0}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm not found in PATH" >&2
  exit 127
fi

if [[ ! -e "$MODEL_PATH" ]]; then
  echo "MODEL_PATH not found: $MODEL_PATH" >&2
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

args=(
  serve
  "$MODEL_PATH"
  --trust-remote-code
  --served-model-name "$SERVED_MODEL_NAME"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --distributed-executor-backend "$DISTRIBUTED_EXECUTOR_BACKEND"
  --enforce-eager
  --dtype "$DTYPE"
)

if [[ -n "$QUANTIZATION" ]]; then
  args+=(--quantization "$QUANTIZATION")
fi

if [[ -n "$LOAD_FORMAT" ]]; then
  args+=(--load-format "$LOAD_FORMAT")
fi

if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]]; then
  if [[ -z "$EXPERT_PARALLEL_FLAG" ]]; then
    if [[ "$AUTO_DETECT_FLAGS" == "1" ]]; then
      help_text="$(vllm_help)"
      EXPERT_PARALLEL_FLAG="$(choose_flag "$help_text" "--enable-expert-parallel" "--enable_expert_parallel")"
    else
      EXPERT_PARALLEL_FLAG="--enable-expert-parallel"
    fi
  fi
  args+=("$EXPERT_PARALLEL_FLAG")
fi

if [[ "$PREFIX_CACHING" == "1" ]]; then
  if [[ "$AUTO_DETECT_FLAGS" == "1" ]]; then
    help_text="$(vllm_help)"
    args+=("$(choose_flag "$help_text" "$ENABLE_PREFIX_CACHING_FLAG" "--enable_prefix_caching")")
  else
    args+=("$ENABLE_PREFIX_CACHING_FLAG")
  fi
else
  if [[ "$AUTO_DETECT_FLAGS" == "1" ]]; then
    help_text="$(vllm_help)"
    args+=("$(choose_flag "$help_text" "$DISABLE_PREFIX_CACHING_FLAG" "--disable-prefix-caching")")
  else
    args+=("$DISABLE_PREFIX_CACHING_FLAG")
  fi
fi

if [[ -n "$HOST" ]]; then
  args+=(--host "$HOST")
fi

if [[ -n "$PORT" ]]; then
  args+=(--port "$PORT")
fi

exec vllm "${args[@]}" "$@"
