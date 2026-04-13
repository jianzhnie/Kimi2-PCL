if [[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export CUDA_DEVICE_MAX_CONNECTIONS=1

LOAD_DIR="${LOAD_DIR:-/llm_workspace_1P/fdd/workspace/MindSpeed-LLM-0227/MindSpeed-LLM/TrainResults/kimi2-base-1T_4k_k8s_mfu33_L32_1024_Arc_Opt2_no_recompute_6144_dies/aea8dbbd-0011-4170-8176-e6c7627132ff}"
SAVE_DIR="${SAVE_DIR:-/llm_workspace_1P/robin/Kimi2-PCL}"

python /llm_workspace_1P/robin/Kimi2-PCL/utils/get_mcore_weights_from_ckpt.py \
    $LOAD_DIR\
    --tp 2 --pp 8 --ep 64 \
    --num-layers 32 \
    --schedules-method dualpipev --vpp-stage 2 \
    --output $SAVE_DIR/model_param_mapping.json




