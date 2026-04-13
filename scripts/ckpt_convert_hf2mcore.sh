#!/usr/bin/env -S env -u BASH_ENV bash
# =============================================================================
# Huggingface (HF) 到 Megatron-Core (MCore) 模型权重转换脚本
# 
# 基于 Kimi2-1T 模型架构配置
# 参考: scripts/pretrain_kimi2_1t_4k.sh
#
# 模型配置:
#   - 32 层 Transformer
#   - Hidden size: 7168
#   - Attention heads: 64 (Q) / 32 (KV) - GQA
#   - MoE: 128 experts, 前 2 层为 Dense
#   - Vocab size: 163840
#
# 默认并行配置 (与训练脚本一致):
#   - TP (Tensor Parallel): 2
#   - PP (Pipeline Parallel): 8  
#   - EP (Expert Parallel): 64
#   - 调度: DualPipeV
# =============================================================================

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

# MOE_TP_EXTEND_EP: 使用 TP 组扩展 EP 并行 (用于昇腾等硬件优化)
# 默认禁用 (0)，可通过 MOE_TP_EXTEND_EP=1 启用
export MOE_TP_EXTEND_EP="${MOE_TP_EXTEND_EP:-0}"

REPO_ROOT="${REPO_ROOT:-"/llm_workspace_1P/robin/Kimi2-PCL"}"

# 检查 REPO_ROOT 是否有效
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "ERROR: REPO_ROOT does not exist: ${REPO_ROOT}" >&2
  exit 1
fi

LOAD_DIR="${LOAD_DIR:-/llm_workspace_1P/robin/hfhub/pcl-kimi2/kimi2-mcore2hf}"
SAVE_DIR="${SAVE_DIR:-/llm_workspace_1P/robin/hfhub/pcl-kimi2/kimi2-hf2mcore_iter900}"

if [[ -z "${LOAD_DIR}" ]]; then
  echo "ERROR: LOAD_DIR must be set (source HuggingFace checkpoint directory)" >&2
  echo "Usage: LOAD_DIR=/path/to/hf/ckpt SAVE_DIR=/path/to/output $0" >&2
  exit 1
fi

if [[ -z "${SAVE_DIR}" ]]; then
  echo "ERROR: SAVE_DIR must be set (target Megatron-Core format output directory)" >&2
  echo "Usage: LOAD_DIR=/path/to/hf/ckpt SAVE_DIR=/path/to/output $0" >&2
  exit 1
fi


# =============================================================================
# 并行配置 (与训练脚本 pretrain_kimi2_1t_4k.sh 保持一致)
# =============================================================================
TP="${TP:-2}"                          # Tensor Parallel size
PP="${PP:-8}"                          # Pipeline Parallel size
EP="${EP:-64}"                         # Expert Parallel size
PP_WORKERS="${PP_WORKERS:-2}"          # PP 并行工作进程数
IO_THREADS="${IO_THREADS:-2}"          # HF 权重加载线程数
SAVE_WORKERS="${SAVE_WORKERS:-0}"      # 保存权重线程数 (0=自动)
CAST_DTYPE="${CAST_DTYPE:-bf16}"       # 输出数据类型
SCHEDULES_METHOD="${SCHEDULES_METHOD:-dualpipev}"  # 调度算法

# =============================================================================
# 模型架构配置 (与 models/config.json 保持一致)
# =============================================================================
NUM_LAYERS="${NUM_LAYERS:-32}"                    # 总层数
FIRST_K_DENSE_REPLACE="${FIRST_K_DENSE_REPLACE:-2}"  # 前 N 层使用 Dense MLP
HIDDEN_SIZE="${HIDDEN_SIZE:-7168}"                # 隐藏层维度
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-18432}"       # Dense FFN 中间维度
MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-12288}"  # MoE FFN 中间维度
VOCAB_SIZE="${VOCAB_SIZE:-163840}"                # 词汇表大小
NUM_EXPERTS="${NUM_EXPERTS:-128}"                 # 专家数量
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-64}"  # Q attention heads
QK_HEAD_DIM="${QK_HEAD_DIM:-128}"                 # QK head 维度
V_HEAD_DIM="${V_HEAD_DIM:-128}"                   # V head 维度
ROTARY_BASE="${ROTARY_BASE:-50000}"               # RoPE 基数
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-131072}"  # 最大位置编码

# 可选配置
NOOP_LAYERS="${NOOP_LAYERS:-}"           # 空层列表 (逗号分隔)
NUM_LAYER_LIST="${NUM_LAYER_LIST:-}"     # 自定义每层分配列表

if [[ ! -d "${LOAD_DIR}" ]]; then
  echo "ERROR: LOAD_DIR does not exist: ${LOAD_DIR}" >&2
  exit 2
fi

# 检查源目录是否包含必要的文件
if [[ ! -f "${LOAD_DIR}/config.json" ]]; then
  echo "WARNING: config.json not found in LOAD_DIR: ${LOAD_DIR}" >&2
  echo "  The conversion may fail if config.json is not present." >&2
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
CONVERT_SCRIPT="${REPO_ROOT}/utils/convert_ckpt_hf2mcore.py"
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
  if [[ "${TP}" -le 1 ]]; then
    echo "MOE_TP_EXTEND_EP=1 需要 TP>1" >&2
    exit 2
  fi
  if (( EP % TP != 0 )); then
    echo "MOE_TP_EXTEND_EP=1 需要 EP 能整除 TP: EP=${EP} TP=${TP}" >&2
    exit 2
  fi
  EXTRA_ARGS+=(--moe-tp-extend-ep)
fi
if [[ -n "${VPP_STAGE:-}" ]]; then
  EXTRA_ARGS+=(--vpp-stage "${VPP_STAGE}")
fi
QK_LAYERNORM="${QK_LAYERNORM:-1}"   # Kimi2-1T 模型启用 QK LayerNorm，默认开启

if [[ -n "${CAST_DTYPE}" ]]; then
  EXTRA_ARGS+=(--cast-dtype "${CAST_DTYPE}")
fi
if [[ "${QK_LAYERNORM:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--qk-layernorm)
fi

echo "Starting conversion..."
echo "  LOAD_DIR: ${LOAD_DIR}"
echo "  SAVE_DIR: ${SAVE_DIR}"
echo "  TP=${TP}, PP=${PP}, EP=${EP}"
echo "  PP_WORKERS=${PP_WORKERS}, SAVE_WORKERS=${SAVE_WORKERS}"
echo "  MOE_TP_EXTEND_EP=${MOE_TP_EXTEND_EP:-0}"
echo ""

python "${CONVERT_SCRIPT}" \
  --load-dir "${LOAD_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --num-layers "${NUM_LAYERS}" \
  --first-k-dense-replace "${FIRST_K_DENSE_REPLACE}" \
  --target-tensor-parallel-size "${TP}" \
  --target-pipeline-parallel-size "${PP}" \
  --target-expert-parallel-size "${EP}" \
  --pp-workers "${PP_WORKERS}" \
  --save-workers "${SAVE_WORKERS}" \
  --hf-io-threads "${IO_THREADS}" \
  --moe-grouped-gemm \
  --rotary-base "${ROTARY_BASE}" \
  --noop-layers "${NOOP_LAYERS}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --ffn-hidden-size "${FFN_HIDDEN_SIZE}" \
  --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE}" \
  --vocab-size "${VOCAB_SIZE}" \
  --num-experts "${NUM_EXPERTS}" \
  --num-attention-heads "${NUM_ATTENTION_HEADS}" \
  --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}" \
  --qk-head-dim "${QK_HEAD_DIM}" \
  --v-head-dim "${V_HEAD_DIM}" \
  --sha256-manifest "${SAVE_DIR}/sha256_manifest.json" \
  "${EXTRA_ARGS[@]}"