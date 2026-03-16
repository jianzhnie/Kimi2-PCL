#!/usr/bin/env bash
source ~/.bashrc

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

REPO_ROOT="/llm_workspace_1P/robin/Kimi2-PCL"
LOAD_DIR="/llm_workspace_1P/fdd/workspace/MindSpeed-LLM-0227/MindSpeed-LLM/TrainResults/kimi2-base-1T_4k_k8s_mfu33_L32_1024_Arc_Opt2_no_recompute_6144_dies/aea8dbbd-0011-4170-8176-e6c7627132ff"
SAVE_DIR="/llm_workspace_1P/robin/hfhub/kimi2-mcore2hf"


TP="${TP:-2}"
PP="${PP:-8}"
EP="${EP:-64}"
NUM_LAYERS="${NUM_LAYERS:-32}"
FIRST_K_DENSE_REPLACE="${FIRST_K_DENSE_REPLACE:-2}"
ROTARY_BASE="${ROTARY_BASE:-50000}"
NOOP_LAYERS="${NOOP_LAYERS:-}"
NUM_LAYER_LIST="${NUM_LAYER_LIST:-}"
SCHEDULES_METHOD="${SCHEDULES_METHOD:-dualpipev}"

HIDDEN_SIZE="${HIDDEN_SIZE:-7168}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-64}"
QK_HEAD_DIM="${QK_HEAD_DIM:-128}"
V_HEAD_DIM="${V_HEAD_DIM:-128}"
QK_POS_EMB_HEAD_DIM="${QK_POS_EMB_HEAD_DIM:-64}"
Q_LORA_RANK="${Q_LORA_RANK:-1536}"

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
if [[ "${MLA_MM_SPLIT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--mla-mm-split)
fi
if [[ -n "${VPP_STAGE:-}" ]]; then
  EXTRA_ARGS+=(--vpp-stage "${VPP_STAGE}")
fi

python "${REPO_ROOT}/utils/convert_ckpt_mcore2hf.py" \
  --load-dir "${LOAD_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --num-layers "${NUM_LAYERS}" \
  --first-k-dense-replace "${FIRST_K_DENSE_REPLACE}" \
  --source-tensor-parallel-size "${TP}" \
  --source-pipeline-parallel-size "${PP}" \
  --source-expert-parallel-size "${EP}" \
  --moe-grouped-gemm \
  --rotary-base "${ROTARY_BASE}" \
  --noop-layers "${NOOP_LAYERS}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --num-experts "${NUM_EXPERTS}" \
  --num-attention-heads "${NUM_ATTENTION_HEADS}" \
  --qk-head-dim "${QK_HEAD_DIM}" \
  --v-head-dim "${V_HEAD_DIM}" \
  --qk-pos-emb-head-dim "${QK_POS_EMB_HEAD_DIM}" \
  --q-lora-rank "${Q_LORA_RANK}" \
  "${EXTRA_ARGS[@]}"
