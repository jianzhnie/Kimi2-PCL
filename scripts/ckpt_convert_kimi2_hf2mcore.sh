#!/usr/bin/env -S env -u BASH_ENV bash
# =============================================================================
# Kimi2-1T-4k Huggingface (HF) 到 Megatron-Core (MCore) 模型权重转换脚本
#
# 使用 ckpt_ckpt_kimi2_1t_4k.py 进行转换
#
# 模型配置:
#   - 32 层 Transformer
#   - Hidden size: 7168
#   - Attention heads: 64 (Q) / 2 (KV) - GQA
#   - MoE: 128 experts, 前 2 层为 Dense
#   - Vocab size: 163840
#
# 支持并行模式:
#   1. DualPipeV: --schedules-method dualpipev
#   2. 标准 VPP: --num-layers-per-virtual-pipeline-stage N
#   3. 纯 PP: 不设 schedules-method 和 vpp-stage
#
# 使用示例:
#   # 模式1: DualPipeV (与训练脚本一致)
#   bash scripts/ckpt_convert_hf2mcore_kimi2.sh
#
#   # 模式2: 标准 VPP (PP=4, 每 vpp stage 4 层)
#   VPP_STAGE=4 PP=4 bash scripts/ckpt_convert_hf2mcore_kimi2.sh
#
#   # 模式3: 纯 PP (PP=8, 无 VPP)
#   SCHEDULES_METHOD="" bash scripts/ckpt_convert_hf2mcore_kimi2.sh
#
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-"/llm_workspace_1P/robin/Kimi2-PCL"}"

# 路径配置
LOAD_DIR="${LOAD_DIR:-/llm_workspace_1P/robin/hfhub/pcl-kimi2/kimi2-mcore2hf}"
SAVE_DIR="${SAVE_DIR:-/llm_workspace_1P/robin/hfhub/pcl-kimi2/kimi2-hf2mcore8}"


if [[ -z "${LOAD_DIR}" ]]; then
  echo "ERROR: LOAD_DIR must be set (source HuggingFace checkpoint directory)" >&2
  echo "Usage: LOAD_DIR=/path/to/hf/ckpt SAVE_DIR=/path/to/output $0" >&2
  exit 1
fi

if [[ -z "${SAVE_DIR}" ]]; then
  echo "ERROR: SAVE_DIR must be set (target Megatron-Core format output directory)" >&2
  exit 1
fi

# 并行配置
TP="${TP:-2}"
PP="${PP:-8}"
EP="${EP:-8}"
VPP_STAGE="${VPP_STAGE:-}"
EXPERT_TP="${EXPERT_TP:-1}"
SCHEDULES_METHOD="${SCHEDULES_METHOD:-}"

# 模型架构配置
NUM_LAYERS="${NUM_LAYERS:-32}"
FIRST_K_DENSE_REPLACE="${FIRST_K_DENSE_REPLACE:-2}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
HIDDEN_SIZE="${HIDDEN_SIZE:-7168}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-64}"
NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-2}"
KV_CHANNELS="${KV_CHANNELS:-128}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-18432}"
MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-12288}"
VOCAB_SIZE="${VOCAB_SIZE:-163840}"
QK_LAYERNORM="${QK_LAYERNORM:-1}"

# 可选配置
MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
NOOP_LAYERS="${NOOP_LAYERS:-}"
NUM_LAYER_LIST="${NUM_LAYER_LIST:-}"
QLORA_NF4="${QLORA_NF4:-0}"

# 检查源目录
if [[ ! -d "${LOAD_DIR}" ]]; then
  echo "ERROR: LOAD_DIR does not exist: ${LOAD_DIR}" >&2
  exit 2
fi

# 创建输出目录
mkdir -p "${SAVE_DIR}"

# 检查转换脚本
CONVERT_SCRIPT="${REPO_ROOT}/utils/convert_kimi2_hf2mcore.py"
if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
  echo "ERROR: Conversion script not found: ${CONVERT_SCRIPT}" >&2
  exit 3
fi

# 确定并行模式
MODE="pure_pp"
if [[ -n "${SCHEDULES_METHOD}" ]]; then
  MODE="dualpipe"
  if [[ -n "${VPP_STAGE}" ]]; then
    echo "ERROR: dualpipev 与 --vpp-stage 不兼容" >&2
    exit 5
  fi
elif [[ -n "${VPP_STAGE}" ]]; then
  MODE="standard_vpp"
fi

echo "==========================================="
echo " HF -> MCore 权重转换 (Kimi2-1T-4k)"
echo "==========================================="
echo "  模式: ${MODE}"
echo "  LOAD_DIR:  ${LOAD_DIR}"
echo "  SAVE_DIR:  ${SAVE_DIR}"
echo "  TP=${TP}, PP=${PP}, EP=${EP}"
if [[ "${MODE}" == "standard_vpp" ]]; then
  echo "  VPP_STAGE=${VPP_STAGE}"
fi
echo "  EXPERT_TP=${EXPERT_TP}"
echo "  NUM_EXPERTS=${NUM_EXPERTS}"
echo "==========================================="
echo ""

# 构建转换参数
EXTRA_ARGS=()
if [[ "${MODE}" == "dualpipe" ]]; then
  EXTRA_ARGS+=(--schedules-method "${SCHEDULES_METHOD}")
fi
if [[ "${MODE}" == "standard_vpp" ]]; then
  EXTRA_ARGS+=(--num-layers-per-virtual-pipeline-stage "${VPP_STAGE}")
fi
if [[ -n "${NUM_LAYER_LIST}" ]]; then
  EXTRA_ARGS+=(--num-layer-list "${NUM_LAYER_LIST}")
fi
if [[ -n "${NOOP_LAYERS}" ]]; then
  EXTRA_ARGS+=(--noop-layers "${NOOP_LAYERS}")
fi
if [[ "${MOE_GROUPED_GEMM}" == "1" ]]; then
  EXTRA_ARGS+=(--moe-grouped-gemm)
fi
if [[ "${QLORA_NF4}" == "1" ]]; then
  EXTRA_ARGS+=(--qlora-nf4)
fi
if [[ "${QK_LAYERNORM}" == "1" ]]; then
  EXTRA_ARGS+=(--qk-layernorm)
fi

python "${CONVERT_SCRIPT}" \
  --load-dir "${LOAD_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --num-layers "${NUM_LAYERS}" \
  --first-k-dense-replace "${FIRST_K_DENSE_REPLACE}" \
  --target-tensor-parallel-size "${TP}" \
  --target-pipeline-parallel-size "${PP}" \
  --target-expert-parallel-size "${EP}" \
  --expert-tensor-parallel-size "${EXPERT_TP}" \
  --num-experts "${NUM_EXPERTS}" \
  "${EXTRA_ARGS[@]}"

echo ""
echo "转换完成! 输出目录: ${SAVE_DIR}"
