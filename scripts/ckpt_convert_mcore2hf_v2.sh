#!/usr/bin/env -S env -u BASH_ENV bash
set -euo pipefail
if [[ -f "${HOME}/.bashrc" ]]; then
  set +u
  source "${HOME}/.bashrc"
  set -u
fi

# 可选的昇腾环境设置（如果存在）
if [[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export CUDA_DEVICE_MAX_CONNECTIONS=1

REPO_ROOT="${REPO_ROOT:-"/llm_workspace_1P/robin/Kimi2-PCL"}"
LOAD_DIR="${LOAD_DIR:-/llm_workspace_1P/fdd/workspace/MindSpeed-LLM-0227/MindSpeed-LLM/TrainResults/kimi2-base-1T_4k_k8s_mfu33_L32_1024_Arc_Opt2_no_recompute_6144_dies/aea8dbbd-0011-4170-8176-e6c7627132ff}"
SAVE_DIR="${SAVE_DIR:-/llm_workspace_1P/robin/hfhub/pcl-kimi2/kimi2-mcore2hf}"

# 检查 REPO_ROOT 是否有效
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "ERROR: REPO_ROOT does not exist: ${REPO_ROOT}" >&2
  exit 1
fi

if [[ -z "${LOAD_DIR}" ]]; then
  echo "ERROR: LOAD_DIR must be set (source Megatron checkpoint directory)" >&2
  echo "Usage: LOAD_DIR=/path/to/mcore/ckpt SAVE_DIR=/path/to/output $0" >&2
  exit 1
fi

if [[ -z "${SAVE_DIR}" ]]; then
  echo "ERROR: SAVE_DIR must be set (target HuggingFace format output directory)" >&2
  echo "Usage: LOAD_DIR=/path/to/mcore/ckpt SAVE_DIR=/path/to/output $0" >&2
  exit 1
fi


TP="${TP:-2}"
PP="${PP:-8}"
EP="${EP:-64}"
PP_WORKERS="${PP_WORKERS:-2}"
IO_THREADS="${IO_THREADS:-8}"
CAST_DTYPE="${CAST_DTYPE:-bf16}"
NUM_LAYERS="${NUM_LAYERS:-32}"
FIRST_K_DENSE_REPLACE="${FIRST_K_DENSE_REPLACE:-2}"
ROTARY_BASE="${ROTARY_BASE:-50000}"
NOOP_LAYERS="${NOOP_LAYERS:-}"
NUM_LAYER_LIST="${NUM_LAYER_LIST:-}"
SCHEDULES_METHOD="${SCHEDULES_METHOD:-dualpipev}"

HIDDEN_SIZE="${HIDDEN_SIZE:-7168}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-18432}"
MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-12288}"
VOCAB_SIZE="${VOCAB_SIZE:-163840}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-64}"
NUM_KEY_VALUE_HEADS="${NUM_KEY_VALUE_HEADS:-32}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-131072}"

if [[ ! -d "${LOAD_DIR}" ]]; then
  echo "ERROR: LOAD_DIR does not exist: ${LOAD_DIR}" >&2
  exit 2
fi

# 创建输出目录（如果不存在）
if [[ ! -d "${SAVE_DIR}" ]]; then
  echo "Creating SAVE_DIR: ${SAVE_DIR}"
  mkdir -p "${SAVE_DIR}" || {
    echo "ERROR: Failed to create SAVE_DIR: ${SAVE_DIR}" >&2
    exit 3
  }
fi

# 检查转换脚本是否存在
CONVERT_SCRIPT="${REPO_ROOT}/utils/convert_ckpt_mcore2hf.py"
if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
  echo "ERROR: Conversion script not found: ${CONVERT_SCRIPT}" >&2
  echo "Please check REPO_ROOT setting (current: ${REPO_ROOT})" >&2
  exit 4
fi

EXTRA_ARGS=()
if [[ -n "${SCHEDULES_METHOD}" ]]; then
  EXTRA_ARGS+=(--schedules-method "${SCHEDULES_METHOD}")
fi
if [[ -n "${NUM_LAYER_LIST}" ]]; then
  EXTRA_ARGS+=(--num-layer-list "${NUM_LAYER_LIST}")
fi
if [[ "${MOE_TP_EXTEND_EP:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--moe-tp-extend-ep)
fi
if [[ -n "${VPP_STAGE:-}" ]]; then
  EXTRA_ARGS+=(--vpp-stage "${VPP_STAGE}")
fi
if [[ -n "${CAST_DTYPE}" ]]; then
  EXTRA_ARGS+=(--cast-dtype "${CAST_DTYPE}")
fi

echo "Starting conversion..."
echo "  LOAD_DIR: ${LOAD_DIR}"
echo "  SAVE_DIR: ${SAVE_DIR}"
echo "  TP=${TP}, PP=${PP}, EP=${EP}"
echo "  NUM_LAYERS=${NUM_LAYERS}, NUM_EXPERTS=${NUM_EXPERTS}"
echo ""

python "${CONVERT_SCRIPT}" \
  --load-dir "${LOAD_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --pp-workers "${PP_WORKERS}" \
  --io-threads "${IO_THREADS}" \
  --num-layers "${NUM_LAYERS}" \
  --first-k-dense-replace "${FIRST_K_DENSE_REPLACE}" \
  --source-tensor-parallel-size "${TP}" \
  --source-pipeline-parallel-size "${PP}" \
  --source-expert-parallel-size "${EP}" \
  --moe-grouped-gemm \
  --rotary-base "${ROTARY_BASE}" \
  --noop-layers "${NOOP_LAYERS}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --ffn-hidden-size "${FFN_HIDDEN_SIZE}" \
  --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE}" \
  --vocab-size "${VOCAB_SIZE}" \
  --num-experts "${NUM_EXPERTS}" \
  --num-attention-heads "${NUM_ATTENTION_HEADS}" \
  --num-key-value-heads "${NUM_KEY_VALUE_HEADS}" \
  --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}" \
  --sha256-manifest "${SAVE_DIR}/sha256_manifest.json" \
  "${EXTRA_ARGS[@]}"
