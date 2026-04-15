if [[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export CUDA_DEVICE_MAX_CONNECTIONS=1


ckpt_path="/llm_workspace_1P/fdd/workspace/MindSpeed-LLM-0227/MindSpeed-LLM/TrainResults/kimi2-base-1T_4k_k8s_mfu33_L32_1024_Arc_Opt2_no_recompute_6144_dies/aea8dbbd-0011-4170-8176-e6c7627132ff"
ckpt_path="/llm_workspace_1P/robin/hfhub/pcl-kimi2/kimi2-hf2mcore"
LOAD_DIR="${LOAD_DIR:-$ckpt_path}"
SAVE_DIR="${SAVE_DIR:-/llm_workspace_1P/robin/Kimi2-PCL}"

# 注意：如果 checkpoint 使用 dualpipev 生成，需要添加 --schedules-method dualpipev
python /llm_workspace_1P/robin/Kimi2-PCL/utils/get_mcore_weights_from_ckpt.py \
    $LOAD_DIR \
    --tp 2 --pp 8 --ep 64 \
    --schedules-method dualpipev \
    --num-layers 32 \
    --output $SAVE_DIR/model_param_mapping_2.json



