# 修改 ascend-toolkit 路径
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


mcore_ckpt=/llm_workspace_1P/robin/hfhub/output/ckpt/k8s_pretrain_qwen3_0point6b_4K_ptd_256_dies_v2
hf_ckpt=/llm_workspace_1P/robin/hfhub/pcl-kimi2/qwen3_0point6b_hf
python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir $mcore_ckpt \
    --save-dir $hf_ckpt \
    --model-type-hf qwen3