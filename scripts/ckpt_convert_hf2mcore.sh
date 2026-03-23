#!/usr/bin/env -S env -u BASH_ENV bash
set -euo pipefail
if [[ -f "${HOME}/.bashrc" ]]; then
  set +u
  source "${HOME}/.bashrc"
  set -u
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

REPO_ROOT="/llm_workspace_1P/robin/Kimi2-PCL"
LOAD_DIR="/llm_workspace_1P/robin/hfhub/kimi2-mcore2hf"
SAVE_DIR="/llm_workspace_1P/robin/hfhub/kimi2-hf2mcore"


TP="${TP:-2}"
PP="${PP:-8}"
EP="${EP:-64}"
PP_WORKERS="${PP_WORKERS:-2}"
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
NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-2}"
QK_HEAD_DIM="${QK_HEAD_DIM:-128}"
V_HEAD_DIM="${V_HEAD_DIM:-128}"
QK_POS_EMB_HEAD_DIM="${QK_POS_EMB_HEAD_DIM:-64}"

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

python "${REPO_ROOT}/utils/convert_ckpt_hf2mcore.py" \
  --load-dir "${LOAD_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --num-layers "${NUM_LAYERS}" \
  --first-k-dense-replace "${FIRST_K_DENSE_REPLACE}" \
  --target-tensor-parallel-size "${TP}" \
  --target-pipeline-parallel-size "${PP}" \
  --target-expert-parallel-size "${EP}" \
  --pp-workers "${PP_WORKERS}" \
  --moe-grouped-gemm \
  --rotary-base "${ROTARY_BASE}" \
  --noop-layers "${NOOP_LAYERS}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --ffn-hidden-size "${FFN_HIDDEN_SIZE}" \
  --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE}" \
  --vocab-size "${VOCAB_SIZE}" \
  --num-experts "${NUM_EXPERTS}" \
  --num-attention-heads "${NUM_ATTENTION_HEADS}" \
  --num-query-groups "${NUM_QUERY_GROUPS}" \
  --qk-head-dim "${QK_HEAD_DIM}" \
  --v-head-dim "${V_HEAD_DIM}" \
  --qk-pos-emb-head-dim "${QK_POS_EMB_HEAD_DIM}" \
  "${EXTRA_ARGS[@]}"
